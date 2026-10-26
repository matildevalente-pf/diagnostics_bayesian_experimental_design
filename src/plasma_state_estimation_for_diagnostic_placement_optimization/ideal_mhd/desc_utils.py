import functools
import math
import warnings
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)

# DESC is imported inside the functions that need it: it is slow to import and pulls
# in its own JAX configuration, and the pure-JAX forward and inverse maps below never
# touch it.
if TYPE_CHECKING:
    import desc.equilibrium
    import desc.profiles

_RHO_FLOOR = 1e-13
"""Smallest `rho` an iterate is evaluated at: the Jacobian is exactly singular on the
axis, and the nudge is far below the achievable real-space accuracy."""

_EXTENDED_RHO_MAX = 2.0
"""Iterate ceiling for `rho` under the beyond-boundary continuation."""

_DAMPING_FACTORS = (0.5, 0.25, 0.125, 0.0625)
"""Reduced steps tried in order when the full step does not lower the residual."""


def evaluate_rz(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rho: Float[NpOrJaxArray, " n_point"],
    theta: Float[NpOrJaxArray, " n_point"],
    zeta: Float[NpOrJaxArray, " n_point"],
    extend_beyond_boundary: bool = False,
) -> tuple[Float[jax.Array, " n_point"], Float[jax.Array, " n_point"]]:
    """Forward map: cylindrical `(R, Z)` of DESC flux coordinates `(rho, theta, zeta)`.

    With `extend_beyond_boundary`, `rho > 1` is continued past the last closed flux
    surface following Attenberger, Houlberg & Hirshman (AHH, 1987, Sec. 2).

    Args:
        coefficients: DESC Fourier-Zernike geometry coefficients.
        rho: Normalized radius per point.
        theta: DESC poloidal angle per point.
        zeta: DESC toroidal angle per point.
        extend_beyond_boundary: Continue the map for `rho > 1` instead of evaluating
            the (there meaningless) Zernike polynomials outside their domain.

    Returns:
        `(R, Z)` in metres.
    """
    geometry = _precontract_geometry(coefficients, zeta, extend_beyond_boundary)
    return _evaluate_rz_precontracted(geometry, rho, theta)


class FluxCoordinates(NamedTuple):
    """Flux coordinates of a set of real-space points, from
    `xyz_to_flux_coordinates`."""

    rho: Float[NpOrJaxArray, " n_point"]
    """Normalized effective radius (`rho = sqrt(s)`), always the final Newton iterate.

    Check `residual <= tol` for convergence. An outside point reaches `rho > 1` with
    extrapolation and parks at `rho = 1` without it; there `rho` is not a flux label.
    """

    theta: Float[NpOrJaxArray, " n_point"]
    """DESC poloidal angle at each point."""

    zeta: Float[NpOrJaxArray, " n_point"]
    """DESC toroidal angle (equal to the cylindrical angle `phi` for these
    equilibria)."""

    residual: Float[NpOrJaxArray, " n_point"]
    """Real-space residual of the Newton root-find, in metres."""


def xyz_to_flux_coordinates(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    points_xyz: Float[NpOrJaxArray, "n_point 3"],
    guess: Float[NpOrJaxArray, "n_point 2"] | None = None,
    tol: float = 1e-12,
    maxiter: int = 50,
    allow_extrapolation: bool = True,
) -> FluxCoordinates:
    """Invert real-space Cartesian points to DESC flux coordinates (Newton root-find).

    A fixed sweep of `maxiter` damped Newton steps, each point frozen once its residual
    drops below `tol`; `rho` is the final iterate, so `residual <= tol` is the test.

    Args:
        coefficients: DESC Fourier-Zernike geometry coefficients defining the flux
            surfaces.
        points_xyz: Cartesian query points.
        guess: Initial `(rho, theta)` per point; the nearest-neighbour `initial_guess`
            is used if None.
        tol: Real-space residual (metres) at which a point is frozen as converged.
        maxiter: Number of Newton iterations; a static Python int, so the sweep stays
            `jax.jit`-traceable and reverse-mode differentiable.
        allow_extrapolation: Continue the map beyond the last closed flux surface (the
            default), returning `rho > 1` for outside points; set False to confine
            iterates to `rho <= 1`, where an outside point parks at `rho = 1`.

    Returns:
        A `FluxCoordinates`, whose converged points are those with `residual <= tol`.
    """
    assert is_stellarator_symmetric(coefficients), (
        "xyz_to_flux_coordinates is only validated for stellarator-symmetric "
        "equilibria. The beyond-boundary continuation carries only the symmetric m=1 "
        "amplitudes (`R ~ cos(theta)` at m=+1, `Z ~ sin(theta)` at m=-1). Supporting "
        "asymmetric equilibria requires also tracking the two extra m=1 amplitudes "
        "(`R ~ sin(theta)` at m=-1, `Z ~ cos(theta)` at m=+1), each paired with its "
        "own angular factor, in the continuation and its Jacobian in "
        "`evaluate_rz`/`evaluate_rz_jacobian`."
    )
    points = jnp.asarray(points_xyz)
    target_r = jnp.hypot(points[:, 0], points[:, 1])
    zeta = jnp.arctan2(points[:, 1], points[:, 0])
    target_z = points[:, 2]

    if guess is None:
        rho, theta = initial_guess(coefficients, points_xyz)
    else:
        guess = jnp.asarray(guess)
        rho, theta = guess[:, 0], guess[:, 1]

    # The toroidal factor is contracted once here, not at every Newton step: `zeta` is
    # fixed for the whole solve, and this also collapses the mode axis from `(l, m, n)`
    # to `(l, m)` (see `_PrecontractedGeometry`).
    geometry = _precontract_geometry(coefficients, zeta, allow_extrapolation)

    # `scan` with a static `length` (not `fori_loop`) is what makes the fixed sweep
    # reverse-mode differentiable: `fori_loop` lowers to a `while_loop`, whose
    # transpose JAX does not implement. That gradient is the unrolled one, costing
    # `O(maxiter)` memory and converging to the implicit-function-theorem gradient as
    # the iterate does; `inverse_mapping.inverse_map_utils.implicit_flux_solve` wraps
    # this solve in that implicit rule, forward and reverse, for the differentiated
    # path.
    state, _ = jax.lax.scan(
        functools.partial(
            _newton_iteration,
            geometry=geometry,
            target_r=target_r,
            target_z=target_z,
            tol=tol,
        ),
        _newton_state(geometry, rho, theta),
        xs=None,
        length=maxiter,
    )
    residual = jnp.hypot(state.r - target_r, state.z - target_z)

    jax.debug.callback(functools.partial(_warn_if_not_converged, tol=tol), residual)
    return FluxCoordinates(
        rho=state.rho,
        theta=state.theta % (2.0 * jnp.pi),
        zeta=zeta,
        residual=residual,
    )


class RZJacobian(NamedTuple):
    """Entries of the 2x2 Jacobian `d(R, Z)/d(rho, theta)` at fixed `zeta`, from
    `evaluate_rz_jacobian`."""

    dr_drho: Float[jax.Array, " n_point"]
    dr_dtheta: Float[jax.Array, " n_point"]
    dz_drho: Float[jax.Array, " n_point"]
    dz_dtheta: Float[jax.Array, " n_point"]


def evaluate_rz_jacobian(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rho: Float[NpOrJaxArray, " n_point"],
    theta: Float[NpOrJaxArray, " n_point"],
    zeta: Float[NpOrJaxArray, " n_point"],
    extend_beyond_boundary: bool = False,
) -> RZJacobian:
    """Analytic `(rho, theta)` partials of the forward map, for the Newton root-find.

    The entries of `[[dR_drho, dR_dtheta], [dZ_drho, dZ_dtheta]]` at fixed `zeta`,
    following the same beyond-boundary continuation as `evaluate_rz`.

    Args:
        coefficients: DESC Fourier-Zernike geometry coefficients.
        rho: Normalized radius per point.
        theta: DESC poloidal angle per point.
        zeta: DESC toroidal angle per point.
        extend_beyond_boundary: Use the beyond-boundary continuation for `rho > 1`.
    """
    geometry = _precontract_geometry(coefficients, zeta, extend_beyond_boundary)
    return _evaluate_rz_and_jacobian_precontracted(geometry, rho, theta)[2]


def initial_guess(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    points_xyz: Float[NpOrJaxArray, "n_point 3"],
) -> tuple[Float[jax.Array, " n_point"], Float[jax.Array, " n_point"]]:
    """Nearest-neighbour initial `(rho, theta)` guess for the inverse root-find.

    Forward-maps a fixed `(rho, theta)` grid at each point's `zeta` and keeps the
    closest node, or a linearized near-axis seed for a point inside the innermost node.

    Args:
        coefficients: DESC Fourier-Zernike geometry coefficients.
        points_xyz: Cartesian query points.

    Returns:
        The `(rho, theta)` initial guess.
    """
    points = jnp.asarray(points_xyz)
    target_r = jnp.hypot(points[:, 0], points[:, 1])
    zeta = jnp.arctan2(points[:, 1], points[:, 0])
    target_z = points[:, 2]
    n_point = points.shape[0]

    # Fixed search grid over the flux disc; `rho` starts above 0 to avoid the axis,
    # where `theta` is degenerate. 48 poloidal nodes keep the nearest node well inside
    # Newton's basin even on strongly shaped surfaces, where the cheaper geometric seed
    # `theta ~ atan2(Z - Z_axis, R - R_axis)` would miss it: that angle matches the DESC
    # flux angle only on near-circular surfaces. DESC seeds its own root-find the same
    # way, with `desc.equilibrium.coords._initial_guess_nn_search` over a 3D grid.
    n_rho, n_theta = 8, 48
    rho_grid = jnp.linspace(1.0 / n_rho, 1.0, n_rho)
    theta_grid = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False)
    node_rho, node_theta = (grid.ravel() for grid in jnp.meshgrid(rho_grid, theta_grid))
    n_node = node_rho.size

    # Every node is evaluated at the query points' own `zeta`, so the toroidal factor is
    # contracted once here rather than inside the loop body.
    geometry = _precontract_geometry(coefficients, zeta, extend_beyond_boundary=False)

    # Scan the grid one node at a time, keeping the running nearest node per query
    # point. Forward-mapping all `n_point * n_node` pairs at once would materialise a
    # `(n_point * n_node, n_group)` intermediate -- many GB for a large batch, enough to
    # OOM a worker -- so the loop caps peak memory at `O(n_point)` while giving the
    # identical nearest-node result.
    initial = (jnp.full(n_point, jnp.inf), jnp.zeros(n_point), jnp.zeros(n_point))
    _, grid_rho, grid_theta = jax.lax.fori_loop(
        0,
        n_node,
        functools.partial(
            _keep_nearest,
            node_rho=node_rho,
            node_theta=node_theta,
            geometry=geometry,
            target_r=target_r,
            target_z=target_z,
            n_point=n_point,
        ),
        initial,
    )

    # Near-axis correction. The grid's innermost node is `rho = 1 / n_rho`, and the map
    # `(rho, theta) -> (R, Z)` degenerates at the axis (`det(J) ~ rho -> 0`), so for a
    # point closer to the axis than that node the nearest-node `theta` is ill-determined
    # and the damped Newton collapses onto `rho ~ 0` and stalls. There the forward map
    # linearizes as `[dR; dZ] ~ M(zeta) @ [rho cos(theta); rho sin(theta)]` with the
    # `m = 1` shaping matrix `M` (well conditioned, unlike `J`); inverting the 2x2 gives
    # a closed-form seed in the correct basin. `M` is read off by finite difference (the
    # axis point plus two points at `rho = axis_probe`), and this seed replaces the grid
    # seed only where it lands inside the innermost grid node -- the region the grid
    # under-resolves. Three extra `evaluate_rz` calls, versus `n_rho * n_theta` for the
    # grid, so the overhead is negligible and there is no second Newton solve.
    axis_probe = 1e-4
    axis_r, axis_z = _evaluate_rz_precontracted(
        geometry, jnp.zeros(n_point), jnp.zeros(n_point)
    )
    probe_r_cos, probe_z_cos = _evaluate_rz_precontracted(
        geometry, jnp.full(n_point, axis_probe), jnp.zeros(n_point)
    )
    probe_r_sin, probe_z_sin = _evaluate_rz_precontracted(
        geometry,
        jnp.full(n_point, axis_probe),
        jnp.full(n_point, 0.5 * jnp.pi),
    )
    shaping_rc = (probe_r_cos - axis_r) / axis_probe
    shaping_rs = (probe_r_sin - axis_r) / axis_probe
    shaping_zc = (probe_z_cos - axis_z) / axis_probe
    shaping_zs = (probe_z_sin - axis_z) / axis_probe
    determinant = shaping_rc * shaping_zs - shaping_rs * shaping_zc
    delta_r = target_r - axis_r
    delta_z = target_z - axis_z
    safe_determinant = jnp.where(jnp.abs(determinant) > 1e-30, determinant, 1.0)
    axis_x = (shaping_zs * delta_r - shaping_rs * delta_z) / safe_determinant
    axis_y = (shaping_rc * delta_z - shaping_zc * delta_r) / safe_determinant
    axis_rho = jnp.hypot(axis_x, axis_y)
    axis_theta = jnp.arctan2(axis_y, axis_x)

    use_axis_seed = (axis_rho < 1.0 / n_rho) & (jnp.abs(determinant) > 1e-30)
    rho = jnp.where(use_axis_seed, axis_rho, grid_rho)
    theta = jnp.where(use_axis_seed, axis_theta, grid_theta)
    return jnp.asarray(rho), jnp.asarray(theta)


def is_stellarator_symmetric(
    coefficients: desc_types.DescFourierZernikeCoefficients,
) -> bool:
    """Whether the `R`/`Z` mode tables carry only stellarator-symmetric parities.

    Under DESC's convention (`fourier` index `>= 0` -> cos, `< 0` -> sin), a symmetric
    equilibrium has `R` even (`m` and `n` share sign) and `Z` odd.
    """
    r_modes = np.asarray(coefficients.r_modes)
    z_modes = np.asarray(coefficients.z_modes)
    r_even = (r_modes[:, 1] >= 0) == (r_modes[:, 2] >= 0)
    z_even = (z_modes[:, 1] >= 0) == (z_modes[:, 2] >= 0)
    return bool(r_even.all() and (~z_even).all())


def desc_equilibrium_to_desc_object(
    desc_equilibrium: desc_types.DescEquilibrium,
    resolution: desc_types.DescResolution,
) -> "desc.equilibrium.Equilibrium":
    """Rebuild a live DESC `Equilibrium` from the serialized geometry and profiles.

    The magnetic axis and the boundary surface are not restored: they only enter DESC
    as boundary conditions of a solve, and nothing here solves.

    Args:
        desc_equilibrium: Serialized equilibrium, geometry plus 1D profiles.
        resolution: Spectral resolution and symmetry of the coefficients.

    Returns:
        The reconstructed `desc.equilibrium.Equilibrium`.
    """
    import desc.equilibrium

    coefficients = desc_equilibrium.coefficients
    # The axis and the boundary surface are not restored: every quantity this folder
    # computes off an equilibrium (`iota`, `<beta>_vol`, `<|F|>_vol` and its
    # normalization) was verified bit-identical without them.
    return desc.equilibrium.Equilibrium(
        R_lmn=np.asarray(coefficients.r_lmn),
        Z_lmn=np.asarray(coefficients.z_lmn),
        L_lmn=np.asarray(coefficients.lambda_lmn),
        L=resolution.radial_resolution,
        M=resolution.poloidal_resolution,
        N=resolution.toroidal_resolution,
        NFP=resolution.n_field_periods,
        sym=resolution.is_stellarator_symmetric,
        spectral_indexing=resolution.spectral_indexing,
        Psi=float(desc_equilibrium.total_toroidal_flux),
        # The stored coefficients are already a converged solution, so DESC must take
        # the boundary exactly as given rather than flipping theta to enforce its own
        # orientation convention.
        check_orientation=False,
        pressure=desc_spline_profile_to_desc_object(desc_equilibrium.pressure_profile),
        current=(
            desc_spline_profile_to_desc_object(desc_equilibrium.current_profile)
            if desc_equilibrium.current_profile is not None
            else None
        ),
        iota=(
            desc_spline_profile_to_desc_object(desc_equilibrium.iota_profile)
            if desc_equilibrium.iota_profile is not None
            else None
        ),
    )


def desc_spline_profile_to_desc_object(
    profile: desc_types.DescSplineProfile,
) -> "desc.profiles.SplineProfile":
    """Rebuild a DESC `SplineProfile` from its serialized form."""
    import desc.profiles

    return desc.profiles.SplineProfile(
        name=profile.name,
        knots=np.asarray(profile.knots),
        values=np.asarray(profile.params),
        method=profile.method,
    )


def fourier_zernike_mode_tables(
    resolution: desc_types.DescResolution,
) -> tuple[
    Int[np.ndarray, "n_modes_r 3"],
    Int[np.ndarray, "n_modes_z 3"],
    Int[np.ndarray, "n_modes_lambda 3"],
]:
    """Build the `(l, m, n)` mode tables of the `R`, `Z` and `lambda` bases.

    Under stellarator symmetry `R` keeps the cosine parity while `Z` and `lambda` keep
    the sine parity, so the three tables differ even at one resolution.

    Args:
        resolution: Spectral resolution and symmetry of the bases.

    Returns:
        The `R`, `Z` and `lambda` mode tables, `n` in per-field-period units.
    """
    if not resolution.is_stellarator_symmetric:
        modes = _basis_mode_table(resolution, False)
        return modes, modes, modes
    return (
        _basis_mode_table(resolution, "cos"),
        _basis_mode_table(resolution, "sin"),
        _basis_mode_table(resolution, "sin"),
    )


def _basis_mode_table(
    resolution: desc_types.DescResolution,
    parity: Literal["cos", "sin"] | bool,
) -> Int[np.ndarray, "n_modes 3"]:
    """Build one Fourier-Zernike mode table at a given parity.

    Args:
        resolution: Spectral resolution of the basis.
        parity: DESC symmetry selector: `"cos"`, `"sin"`, or `False` for no symmetry.

    Returns:
        The `(l, m, n)` mode table.
    """
    import desc.basis

    return np.asarray(
        desc.basis.FourierZernikeBasis(
            L=resolution.radial_resolution,
            M=resolution.poloidal_resolution,
            N=resolution.toroidal_resolution,
            NFP=resolution.n_field_periods,
            sym=parity,  # type: ignore[arg-type]
            spectral_indexing=resolution.spectral_indexing,
        ).modes,
        dtype=int,
    )


def _zernike_polynomial_coefficients(
    radial_mode_numbers: Int[NpOrJaxArray, " n_mode"],
    poloidal_mode_numbers: Int[NpOrJaxArray, " n_mode"],
) -> Float[np.ndarray, "n_power n_mode"]:
    """Exact integer coefficients of `R_l^|m|(rho)` as a polynomial in `rho`.

    Column `j` holds mode `j`, so `R_l^|m|(rho) = sum_i coefficients[i, j] * rho^i` and
    a radial basis evaluates as one matmul against a Vandermonde of `rho` powers.

    Args:
        radial_mode_numbers: Zernike radial mode numbers `l`, one per mode.
        poloidal_mode_numbers: Poloidal mode numbers `m`, one per mode.

    Returns:
        Polynomial coefficient table of shape `(n_power, n_mode)`.
    """
    radial = np.asarray(radial_mode_numbers, dtype=int)
    poloidal = np.abs(np.asarray(poloidal_mode_numbers, dtype=int))
    if np.any(radial < poloidal) or np.any((radial - poloidal) % 2 != 0):
        raise ValueError(
            "Zernike modes require l >= |m| and l - |m| even; got invalid (l, m) pairs."
        )

    n_power = int(radial.max()) + 1 if radial.size else 1
    coefficients = np.zeros((n_power, radial.size))
    for j, (l_mode, m_mode) in enumerate(zip(radial, poloidal)):
        for k in range((l_mode - m_mode) // 2 + 1):
            coefficients[l_mode - 2 * k, j] += (-1) ** k * (
                math.factorial(l_mode - k)
                // (
                    math.factorial(k)
                    * math.factorial((l_mode + m_mode) // 2 - k)
                    * math.factorial((l_mode - m_mode) // 2 - k)
                )
            )
    return coefficients


class _FluxBasisTables(NamedTuple):
    """Static tables of one Fourier-Zernike mode table.

    These depend on the `(l, m, n)` mode numbers alone, so they are built once per mode
    table (cached) and are compile-time constants inside a `jax.jit` trace.
    """

    group_index: Int[np.ndarray, " n_mode"]
    """`(l, m)` group of each mode `(l, m, n)`."""

    toroidal_slot: Int[np.ndarray, " n_mode"]
    """Column of the toroidal basis each mode reads: `cos(|n| NFP zeta)` in slot `|n|`,
    `sin(|n| NFP zeta)` in slot `n_max + |n|`."""

    n_toroidal_slot: int
    """`2 n_max + 1` columns of the toroidal basis."""

    poloidal_mode_numbers: Int[np.ndarray, " n_group"]
    """Signed `m` per `(l, m)` group."""

    radial_polynomial: Float[np.ndarray, "n_power n_group"]
    """`R_l^|m|(rho)` of each group, as coefficients of `rho^i`."""

    radial_polynomial_d_rho: Float[np.ndarray, "n_power n_group"]
    """`d/drho` of `radial_polynomial`, in the same `rho^i` basis."""


class _PrecontractedGeometry(NamedTuple):
    """A geometry whose toroidal factor is already contracted at fixed `zeta`.

    Contracting `sum_n c_lmn F_n(NFP * zeta)` once per call collapses the mode axis
    from `(l, m, n)` to `(l, m)`, which is where most of the saving comes from.
    """

    r_amplitudes: Float[jax.Array, "n_point n_group"]
    """Per-point `(l, m)` amplitudes of `R`."""

    z_amplitudes: Float[jax.Array, "n_point n_group"]
    """Per-point `(l, m)` amplitudes of `Z`."""

    r_tables: _FluxBasisTables
    z_tables: _FluxBasisTables

    boundary_amplitude_r: Float[jax.Array, ""]
    boundary_amplitude_z: Float[jax.Array, ""]
    """Boundary `m = 1` amplitudes driving the beyond-boundary continuation."""

    extend_beyond_boundary: bool
    """Whether the continuation is active (static)."""


def _flux_basis_tables(modes: Int[NpOrJaxArray, "n_mode 3"]) -> _FluxBasisTables:
    """Static tables of a mode table, cached on the mode numbers."""
    modes = np.asarray(modes, dtype=int)
    return _cached_flux_basis_tables(modes.tobytes(), modes.shape)


@functools.lru_cache(maxsize=16)
def _cached_flux_basis_tables(
    modes_bytes: bytes, shape: tuple[int, int]
) -> _FluxBasisTables:
    modes = np.frombuffer(modes_bytes, dtype=int).reshape(shape)
    groups, group_index = np.unique(modes[:, :2], axis=0, return_inverse=True)
    toroidal = modes[:, 2]
    n_max = int(np.abs(toroidal).max())

    polynomial = _zernike_polynomial_coefficients(groups[:, 0], groups[:, 1])
    derivative = np.zeros_like(polynomial)
    derivative[:-1] = (np.arange(polynomial.shape[0])[:, None] * polynomial)[1:]
    return _FluxBasisTables(
        group_index=np.reshape(group_index, -1),
        toroidal_slot=np.where(
            toroidal >= 0, np.abs(toroidal), n_max + np.abs(toroidal)
        ),
        n_toroidal_slot=2 * n_max + 1,
        poloidal_mode_numbers=np.ascontiguousarray(groups[:, 1]),
        radial_polynomial=polynomial,
        radial_polynomial_d_rho=derivative,
    )


def _precontract_geometry(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    zeta: Float[NpOrJaxArray, " n_point"],
    extend_beyond_boundary: bool,
) -> _PrecontractedGeometry:
    """Contract the toroidal factor of `R` and `Z` at the query points' `zeta`."""
    r_tables = _flux_basis_tables(coefficients.r_modes)
    z_tables = _flux_basis_tables(coefficients.z_modes)
    # The boundary `(m, n) = (1, 0)` Fourier amplitudes of `R` (cos) and `Z` (sin)
    # drive the beyond-boundary continuation of Attenberger, Houlberg & Hirshman: for
    # `rho > 1` they grow proportionally to `rho` while all other coefficients stay
    # frozen. At `rho = 1` every Zernike radial factor is one (`R_l^|m|(1) = 1`), so
    # each amplitude is the plain sum of the coefficients of all radial modes with
    # the matching `(m, n)`.
    r_modes = jnp.asarray(coefficients.r_modes)
    z_modes = jnp.asarray(coefficients.z_modes)
    r_mask = (r_modes[:, 1] == 1) & (r_modes[:, 2] == 0)
    z_mask = (z_modes[:, 1] == -1) & (z_modes[:, 2] == 0)
    amplitude_r = jnp.sum(jnp.where(r_mask, jnp.asarray(coefficients.r_lmn), 0.0))
    amplitude_z = jnp.sum(jnp.where(z_mask, jnp.asarray(coefficients.z_lmn), 0.0))
    n_field_periods = coefficients.n_field_periods
    return _PrecontractedGeometry(
        r_amplitudes=_toroidal_contraction(
            r_tables, coefficients.r_lmn, zeta, n_field_periods
        ),
        z_amplitudes=_toroidal_contraction(
            z_tables, coefficients.z_lmn, zeta, n_field_periods
        ),
        r_tables=r_tables,
        z_tables=z_tables,
        boundary_amplitude_r=amplitude_r,
        boundary_amplitude_z=amplitude_z,
        extend_beyond_boundary=extend_beyond_boundary,
    )


def _toroidal_contraction(
    tables: _FluxBasisTables,
    coefficients: Float[NpOrJaxArray, " n_mode"],
    zeta: Float[NpOrJaxArray, " n_point"],
    n_field_periods: int,
) -> Float[jax.Array, "n_point n_group"]:
    """`sum_n c_lmn F_n(NFP * zeta)` per point and `(l, m)` group: one small matmul of
    `cos`/`sin` of the distinct `|n|` against the coefficients laid out by toroidal slot
    and group. Linear in `coefficients`, so reverse-mode gradients flow through."""
    n_group = tables.poloidal_mode_numbers.shape[0]
    table = (
        jnp.zeros((tables.n_toroidal_slot, n_group))
        .at[tables.toroidal_slot, tables.group_index]
        .set(jnp.asarray(coefficients))
    )
    n_max = (tables.n_toroidal_slot - 1) // 2
    angle = n_field_periods * jnp.arange(n_max + 1) * jnp.asarray(zeta)[:, None]
    basis = jnp.concatenate([jnp.cos(angle), jnp.sin(angle[:, 1:])], axis=1)
    return basis @ table


def _rho_powers(
    rho: Float[NpOrJaxArray, "..."], n_power: int
) -> Float[jax.Array, "... n_power"]:
    """`[1, rho, rho^2, ...]` along a new trailing axis.

    Built by repeated multiplication: `rho ** arange(n_power)` has a NaN reverse-mode
    derivative at `rho = 0`, and the magnetic axis is a real query point.
    """
    rho = jnp.asarray(rho)
    columns = [jnp.ones_like(rho)]
    for _ in range(n_power - 1):
        columns.append(columns[-1] * rho)
    return jnp.stack(columns, axis=-1)


def _radial_factor(
    polynomial: Float[np.ndarray, "n_power n_group"],
    powers: Float[jax.Array, "... n_power_max"],
) -> Float[jax.Array, "... n_group"]:
    return powers[..., : polynomial.shape[0]] @ jnp.asarray(polynomial)


def _poloidal_factors(
    poloidal_mode_numbers: Int[np.ndarray, " n_group"],
    theta: Float[NpOrJaxArray, "..."],
) -> tuple[Float[jax.Array, "... n_group"], Float[jax.Array, "... n_group"]]:
    """`F_m(theta)` and its `theta` derivative, from `cos`/`sin` of the few distinct
    `|m|` that every group then reads its own column of."""
    poloidal_modes = np.asarray(poloidal_mode_numbers)
    order = np.abs(poloidal_modes)
    angle = jnp.arange(int(order.max()) + 1) * jnp.asarray(theta)[..., None]
    cosine = jnp.take(jnp.cos(angle), order, axis=-1)
    sine = jnp.take(jnp.sin(angle), order, axis=-1)
    is_cosine = poloidal_modes >= 0
    return (
        jnp.where(is_cosine, cosine, sine),
        jnp.asarray(order) * jnp.where(is_cosine, -sine, cosine),
    )


def _evaluate_rz_precontracted(
    geometry: _PrecontractedGeometry,
    rho: Float[NpOrJaxArray, "... n_point"],
    theta: Float[NpOrJaxArray, "... n_point"],
) -> tuple[Float[jax.Array, "... n_point"], Float[jax.Array, "... n_point"]]:
    """`(R, Z)` of a precontracted geometry; leading batch axes are broadcast."""
    rho, theta = jnp.asarray(rho), jnp.asarray(theta)
    rho_inside = jnp.minimum(rho, 1.0) if geometry.extend_beyond_boundary else rho
    # Vandermonde width shared by the `R` and `Z` radial tables.
    n_power = max(
        geometry.r_tables.radial_polynomial.shape[0],
        geometry.z_tables.radial_polynomial.shape[0],
    )
    powers = _rho_powers(rho_inside, n_power)
    poloidal_r, _ = _poloidal_factors(geometry.r_tables.poloidal_mode_numbers, theta)
    poloidal_z, _ = _poloidal_factors(geometry.z_tables.poloidal_mode_numbers, theta)
    r = jnp.sum(
        geometry.r_amplitudes
        * _radial_factor(geometry.r_tables.radial_polynomial, powers)
        * poloidal_r,
        axis=-1,
    )
    z = jnp.sum(
        geometry.z_amplitudes
        * _radial_factor(geometry.z_tables.radial_polynomial, powers)
        * poloidal_z,
        axis=-1,
    )
    if not geometry.extend_beyond_boundary:
        return r, z
    excess = jnp.maximum(rho - 1.0, 0.0)
    return (
        r + excess * geometry.boundary_amplitude_r * jnp.cos(theta),
        z + excess * geometry.boundary_amplitude_z * jnp.sin(theta),
    )


def _evaluate_rz_and_jacobian_precontracted(
    geometry: _PrecontractedGeometry,
    rho: Float[NpOrJaxArray, "... n_point"],
    theta: Float[NpOrJaxArray, "... n_point"],
) -> tuple[
    Float[jax.Array, "... n_point"], Float[jax.Array, "... n_point"], RZJacobian
]:
    """`(R, Z)` and the 2x2 `(rho, theta)` Jacobian from one shared basis build.

    The value and both partials reuse the same Vandermonde and the same `cos`/`sin`
    pair, so six outputs cost five basis matrices instead of eighteen.
    """
    rho, theta = jnp.asarray(rho), jnp.asarray(theta)
    rho_inside = jnp.minimum(rho, 1.0) if geometry.extend_beyond_boundary else rho
    # Vandermonde width shared by the `R` and `Z` radial tables.
    n_power = max(
        geometry.r_tables.radial_polynomial.shape[0],
        geometry.z_tables.radial_polynomial.shape[0],
    )
    powers = _rho_powers(rho_inside, n_power)
    r, dr_drho, dr_dtheta = _series_and_partials(
        geometry.r_tables, geometry.r_amplitudes, powers, theta
    )
    z, dz_drho, dz_dtheta = _series_and_partials(
        geometry.z_tables, geometry.z_amplitudes, powers, theta
    )
    if not geometry.extend_beyond_boundary:
        return (
            r,
            z,
            RZJacobian(
                dr_drho=dr_drho,
                dr_dtheta=dr_dtheta,
                dz_drho=dz_drho,
                dz_dtheta=dz_dtheta,
            ),
        )

    outside = rho > 1.0
    excess = jnp.maximum(rho - 1.0, 0.0)
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    amplitude_r = geometry.boundary_amplitude_r
    amplitude_z = geometry.boundary_amplitude_z
    return (
        r + excess * amplitude_r * cosine,
        z + excess * amplitude_z * sine,
        RZJacobian(
            dr_drho=jnp.where(outside, amplitude_r * cosine, dr_drho),
            dr_dtheta=dr_dtheta - excess * amplitude_r * sine,
            dz_drho=jnp.where(outside, amplitude_z * sine, dz_drho),
            dz_dtheta=dz_dtheta + excess * amplitude_z * cosine,
        ),
    )


def _series_and_partials(
    tables: _FluxBasisTables,
    amplitudes: Float[jax.Array, "n_point n_group"],
    powers: Float[jax.Array, "... n_power"],
    theta: Float[NpOrJaxArray, "... n_point"],
) -> tuple[
    Float[jax.Array, "... n_point"],
    Float[jax.Array, "... n_point"],
    Float[jax.Array, "... n_point"],
]:
    """One precontracted series and its `(rho, theta)` partials, sharing every
    factor."""
    radial = _radial_factor(tables.radial_polynomial, powers)
    radial_d_rho = _radial_factor(tables.radial_polynomial_d_rho, powers)
    poloidal, poloidal_d_theta = _poloidal_factors(tables.poloidal_mode_numbers, theta)
    weighted = amplitudes * radial
    return (
        jnp.sum(weighted * poloidal, axis=-1),
        jnp.sum(amplitudes * radial_d_rho * poloidal, axis=-1),
        jnp.sum(weighted * poloidal_d_theta, axis=-1),
    )


class _NewtonState(NamedTuple):
    """A Newton iterate with the forward map and its Jacobian already evaluated there,
    so the next step starts without a basis build."""

    rho: Float[jax.Array, " n_point"]
    theta: Float[jax.Array, " n_point"]
    r: Float[jax.Array, " n_point"]
    z: Float[jax.Array, " n_point"]
    jacobian: RZJacobian


def _newton_state(
    geometry: _PrecontractedGeometry,
    rho: Float[NpOrJaxArray, " n_point"],
    theta: Float[NpOrJaxArray, " n_point"],
) -> _NewtonState:
    """The state of an iterate, nudged off the axis where the Jacobian is singular."""
    rho = jnp.maximum(jnp.asarray(rho), _RHO_FLOOR)
    theta = jnp.asarray(theta)
    r, z, jacobian = _evaluate_rz_and_jacobian_precontracted(geometry, rho, theta)
    return _NewtonState(rho=rho, theta=theta, r=r, z=z, jacobian=jacobian)


def _newton_iteration(
    state: _NewtonState,
    _: None,
    geometry: _PrecontractedGeometry,
    target_r: Float[NpOrJaxArray, " n_point"],
    target_z: Float[NpOrJaxArray, " n_point"],
    tol: float,
) -> tuple[_NewtonState, None]:
    """One `lax.scan` step of the Newton sweep; a no-op once every point has converged.

    Freezing replaces the data-dependent early stop of a `while_loop`: it keeps `tol`
    meaningful and avoids re-stepping converged points.
    """
    residual = jnp.hypot(state.r - target_r, state.z - target_z)
    active = residual > tol
    state = jax.lax.cond(
        jnp.any(active),
        functools.partial(
            _newton_step, geometry=geometry, target_r=target_r, target_z=target_z
        ),
        _unchanged,
        state,
        active,
        residual,
    )
    return state, None


def _unchanged(
    state: _NewtonState,
    active: Float[jax.Array, " n_point"],
    residual: Float[jax.Array, " n_point"],
) -> _NewtonState:
    return state


def _newton_step(
    state: _NewtonState,
    active: Float[jax.Array, " n_point"],
    residual: Float[jax.Array, " n_point"],
    geometry: _PrecontractedGeometry,
    target_r: Float[NpOrJaxArray, " n_point"],
    target_z: Float[NpOrJaxArray, " n_point"],
) -> _NewtonState:
    """One damped Newton iteration on `(rho, theta)`: the full step (AHH Eq. 3) where it
    lowers the residual, and the step-halving ladder (`_damped_candidates`) only when
    some active point is not improved.

    Args:
        state: Current iterate with its forward map and Jacobian.
        active: Points still above the tolerance; the others keep their iterate.
        residual: Real-space residual of the current iterate.
        geometry: Geometry with the toroidal factor already contracted at the query
            points' `zeta`, carrying whether the continuation is active.
        target_r: Target cylindrical `R`.
        target_z: Target cylindrical `Z`.

    Returns:
        The updated state.
    """
    rho_ceiling = _EXTENDED_RHO_MAX if geometry.extend_beyond_boundary else 1.0
    jacobian = state.jacobian
    residual_r = state.r - target_r
    residual_z = state.z - target_z
    determinant = (
        jacobian.dr_drho * jacobian.dz_dtheta - jacobian.dr_dtheta * jacobian.dz_drho
    )
    numerator_rho = jacobian.dz_dtheta * residual_r - jacobian.dr_dtheta * residual_z
    numerator_theta = jacobian.dr_drho * residual_z - jacobian.dz_drho * residual_r

    # `step_theta` is capped to +-pi: any angle is within pi of the iterate by
    # periodicity, and an uncapped step near the degenerate axis (`determinant ~ rho`)
    # reaches `|theta| ~ 1e14`, where the float64 ULP exceeds the correction scale and
    # freezes `theta`. The `rho` step needs no cap -- the fold/clip fixups bound every
    # candidate.
    full = _newton_state(
        geometry,
        *_fixed_up_candidate(
            state.rho,
            state.theta,
            numerator_rho / determinant,
            jnp.clip(numerator_theta / determinant, -jnp.pi, jnp.pi),
            rho_ceiling,
        ),
    )
    improves = jnp.hypot(full.r - target_r, full.z - target_z) < residual

    chosen = jax.lax.cond(
        jnp.any(active & ~improves),
        functools.partial(
            _damped_candidates,
            geometry=geometry,
            target_r=target_r,
            target_z=target_z,
            rho_ceiling=rho_ceiling,
        ),
        _full_step,
        state,
        full,
        improves,
        residual,
        numerator_rho,
        numerator_theta,
        determinant,
    )
    return jax.tree.map(functools.partial(jnp.where, active), chosen, state)


def _full_step(
    state: _NewtonState,
    full: _NewtonState,
    improves: Float[jax.Array, " n_point"],
    residual: Float[jax.Array, " n_point"],
    numerator_rho: Float[jax.Array, " n_point"],
    numerator_theta: Float[jax.Array, " n_point"],
    determinant: Float[jax.Array, " n_point"],
) -> _NewtonState:
    return full


def _damped_candidates(
    state: _NewtonState,
    full: _NewtonState,
    improves: Float[jax.Array, " n_point"],
    residual: Float[jax.Array, " n_point"],
    numerator_rho: Float[jax.Array, " n_point"],
    numerator_theta: Float[jax.Array, " n_point"],
    determinant: Float[jax.Array, " n_point"],
    geometry: _PrecontractedGeometry,
    target_r: Float[NpOrJaxArray, " n_point"],
    target_z: Float[NpOrJaxArray, " n_point"],
    rho_ceiling: float,
) -> _NewtonState:
    """The step-halving ladder of AHH: per point the largest reduced step that lowers
    the residual, the smallest when none does, so the iterate keeps moving; returns the
    chosen iterate with its forward map and Jacobian evaluated."""
    next_jacobian = full.jacobian
    determinant_next = (
        next_jacobian.dr_drho * next_jacobian.dz_dtheta
        - next_jacobian.dr_dtheta * next_jacobian.dz_drho
    )
    # Reduced steps divide by the Jacobian blend of AHH Eq. 5, `0.75 det(k) + 0.25
    # det(k+1)` with `det(k+1)` at the full-step candidate: this carries the iteration
    # across the `det` discontinuity at `rho = 1` of the beyond-boundary continuation.
    determinant_blend = 0.75 * determinant + 0.25 * determinant_next
    damping = jnp.asarray(_DAMPING_FACTORS)[:, None]
    candidate_rho, candidate_theta = _fixed_up_candidate(
        state.rho[None, :],
        state.theta[None, :],
        damping * (numerator_rho / determinant_blend)[None, :],
        damping
        * jnp.clip(numerator_theta / determinant_blend, -jnp.pi, jnp.pi)[None, :],
        rho_ceiling,
    )
    # The precontracted amplitudes are per point, so the `(n_damping, n_point)`
    # candidate grid broadcasts against them directly.
    candidate_r, candidate_z = _evaluate_rz_precontracted(
        geometry, candidate_rho, candidate_theta
    )
    candidate_improves = (
        jnp.hypot(candidate_r - target_r[None, :], candidate_z - target_z[None, :])
        < residual[None, :]
    )

    # The full step leads the ladder, so `argmax` over the improvement matrix returns
    # the first (largest) improving factor, exactly what the sequential halving loop
    # would accept; the last rung is taken when none improves.
    ladder_rho = jnp.concatenate([full.rho[None, :], candidate_rho])
    ladder_theta = jnp.concatenate([full.theta[None, :], candidate_theta])
    ladder_improves = jnp.concatenate([improves[None, :], candidate_improves])
    chosen = jnp.where(
        jnp.any(ladder_improves, axis=0),
        jnp.argmax(ladder_improves, axis=0),
        ladder_rho.shape[0] - 1,
    )
    point_index = jnp.arange(chosen.shape[0])
    return _newton_state(
        geometry, ladder_rho[chosen, point_index], ladder_theta[chosen, point_index]
    )


def _fixed_up_candidate(
    rho: Float[jax.Array, "... n_point"],
    theta: Float[jax.Array, "... n_point"],
    step_rho: Float[jax.Array, "... n_point"],
    step_theta: Float[jax.Array, "... n_point"],
    rho_ceiling: float,
) -> tuple[Float[jax.Array, "... n_point"], Float[jax.Array, "... n_point"]]:
    """A Newton candidate after the standard fixups: flip `theta` by `pi` when `rho`
    steps negative, wrap `theta`, clip `rho` to its range, and keep the current iterate
    for a non-finite step (degenerate Jacobian on the magnetic axis)."""
    candidate_rho = rho - step_rho
    candidate_theta = jnp.where(
        candidate_rho < 0, theta - step_theta + jnp.pi, theta - step_theta
    )
    candidate_theta = candidate_theta % (2.0 * jnp.pi)
    candidate_rho = jnp.clip(jnp.abs(candidate_rho), 0.0, rho_ceiling)
    finite = jnp.isfinite(candidate_rho) & jnp.isfinite(candidate_theta)
    return (
        jnp.where(finite, candidate_rho, rho),
        jnp.where(finite, candidate_theta, theta),
    )


def _warn_if_not_converged(residual: np.ndarray, tol: float) -> None:
    """Warn once when any point's residual exceeds the tolerance.

    Static message (no interpolated residual value) so `warnings` de-duplicates it to a
    single line even when this runs thousands of times under MCMC.
    """
    if bool(np.any(np.asarray(residual) > tol)):
        warnings.warn(
            f"xyz_to_flux_coordinates did not converge for some points: "
            f"real-space residual exceeds tol={tol:.1e} m. Increase maxiter, or "
            f"check the query points lie inside the plasma.",
            stacklevel=2,
        )


def _keep_nearest(
    node: int,
    carry: tuple[jax.Array, jax.Array, jax.Array],
    node_rho: Float[jax.Array, " n_node"],
    node_theta: Float[jax.Array, " n_node"],
    geometry: _PrecontractedGeometry,
    target_r: Float[NpOrJaxArray, " n_point"],
    target_z: Float[NpOrJaxArray, " n_point"],
    n_point: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One `fori_loop` step of the nearest-node grid scan of `initial_guess`."""
    best_distance, best_rho, best_theta = carry
    rho_node = node_rho[node]
    theta_node = node_theta[node]
    r_node, z_node = _evaluate_rz_precontracted(
        geometry,
        jnp.full(n_point, rho_node),
        jnp.full(n_point, theta_node),
    )
    distance = jnp.hypot(r_node - target_r, z_node - target_z)
    closer = distance < best_distance
    return (
        jnp.where(closer, distance, best_distance),
        jnp.where(closer, rho_node, best_rho),
        jnp.where(closer, theta_node, best_theta),
    )
