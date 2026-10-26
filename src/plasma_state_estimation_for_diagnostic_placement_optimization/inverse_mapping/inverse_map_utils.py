# Differentiable xyz to flux-coordinate inverse map: the Newton solve of
# `ideal_mhd.desc_utils` under an implicit-function-theorem JVP, forward and reverse.

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)

_Solve = Callable[
    [jax.Array, jax.Array, jax.Array],
    tuple[jt.Float[jax.Array, " n_point"], jt.Float[jax.Array, " n_point"]],
]
"""`(r_lmn, z_lmn, points) -> (rho, theta)`, the solve with its derivative rule."""


def implicit_flux_solve(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    points_xyz: jt.Float[NpOrJaxArray, "n_point 3"],
    guess: jt.Float[NpOrJaxArray, "n_point 2"] | None = None,
    tol: float = 1e-12,
    maxiter: int = 50,
) -> tuple[jt.Float[jax.Array, " n_point"], jt.Float[jax.Array, " n_point"]]:
    """`(rho, theta)` of the inverse map, differentiable in forward and reverse mode by
    the implicit function theorem at the converged root, at a cost independent of
    `maxiter`; `r_lmn`, `z_lmn` and `points_xyz` all carry derivatives.

    Args:
        coefficients: DESC Fourier-Zernike geometry coefficients defining the surfaces.
        points_xyz: Cartesian query points.
        guess: Initial `(rho, theta)` per point; the nearest-neighbour
            `desc_utils.initial_guess` is used if None. For fixed diagnostic points,
            precompute it once and reuse it across objective evaluations.
        tol: Real-space convergence tolerance of the Newton solve in metres.
        maxiter: Number of Newton iterations (static Python int).

    Returns:
        The `(rho, theta)` arrays of the converged flux coordinates.
    """
    # The custom_jvp is built per call, binding what stays fixed with
    # `functools.partial` rather than with `nondiff_argnums`: those must be hashable,
    # and the geometry's mode tables must stay concrete inside the trace.
    fixed = _FixedInputs(
        coefficients=coefficients, guess=guess, tol=tol, maxiter=maxiter
    )
    solve = jax.custom_jvp(functools.partial(_solve, fixed))
    solve.defjvp(functools.partial(_solve_jvp, fixed, solve))
    return solve(coefficients.r_lmn, coefficients.z_lmn, jnp.asarray(points_xyz))


class _FixedInputs(NamedTuple):
    """What the solve holds fixed while coefficients and points are differentiated."""

    coefficients: desc_types.DescFourierZernikeCoefficients
    guess: jt.Float[NpOrJaxArray, "n_point 2"] | None
    tol: float
    maxiter: int


def _solve(
    fixed: _FixedInputs,
    r_lmn: jax.Array,
    z_lmn: jax.Array,
    points: jt.Float[jax.Array, "n_point 3"],
) -> tuple[jt.Float[jax.Array, " n_point"], jt.Float[jax.Array, " n_point"]]:
    """Primal pass: the unmodified Newton solve at the given geometry and points."""
    flux = desc_utils.xyz_to_flux_coordinates(
        _geometry(fixed, r_lmn, z_lmn),
        points,
        guess=fixed.guess,
        tol=fixed.tol,
        maxiter=fixed.maxiter,
    )
    return jnp.asarray(flux.rho), jnp.asarray(flux.theta)


def _solve_jvp(
    fixed: _FixedInputs,
    solve: _Solve,
    primals: tuple[jax.Array, jax.Array, jax.Array],
    tangents: tuple[jax.Array, jax.Array, jax.Array],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Implicit-function-theorem JVP: at the root of `F(rho, theta; c, x) = (R, Z)(rho,
    theta, zeta(x); c) - (R, Z)(x)`, `[rho_dot, theta_dot] = -J^-1 F_dot` with
    `J = d(R, Z)/d(rho, theta)`; one per-point 2x2 solve on the tangent of `F`."""
    # Blondel et al., https://arxiv.org/abs/2105.15183, Sec. 2.1, Eq. 2. The rule is
    # linear in the tangents, so JAX derives reverse mode by transposing it (JAX docs,
    # `jax.custom_jvp`), and one rule serves both modes.
    r_lmn, z_lmn, points = primals
    r_dot, z_dot, points_dot = tangents
    # The primal goes through `solve` itself, so a second derivative differentiates
    # this rule rather than the Newton scan.
    rho, theta = solve(r_lmn, z_lmn, points)
    (_, _, zeta), (target_r_dot, target_z_dot, zeta_dot) = jax.jvp(
        _cylindrical, (points,), (points_dot,)
    )
    _, (forward_r_dot, forward_z_dot) = jax.jvp(
        functools.partial(_forward_rz, fixed, rho, theta),
        (r_lmn, z_lmn, zeta),
        (r_dot, z_dot, zeta_dot),
    )
    residual_r_dot = forward_r_dot - target_r_dot
    residual_z_dot = forward_z_dot - target_z_dot
    jacobian = desc_utils.evaluate_rz_jacobian(
        _geometry(fixed, r_lmn, z_lmn), rho, theta, zeta, extend_beyond_boundary=True
    )
    determinant = (
        jacobian.dr_drho * jacobian.dz_dtheta - jacobian.dr_dtheta * jacobian.dz_drho
    )
    # Degenerate points (exactly on the axis / parked outside) get zero tangent
    # instead of inf; their likelihood contribution is masked anyway.
    safe_determinant = jnp.where(jnp.abs(determinant) > 1e-30, determinant, jnp.inf)
    rho_dot = (
        -(jacobian.dz_dtheta * residual_r_dot - jacobian.dr_dtheta * residual_z_dot)
        / safe_determinant
    )
    theta_dot = (
        -(jacobian.dr_drho * residual_z_dot - jacobian.dz_drho * residual_r_dot)
        / safe_determinant
    )
    return (rho, theta), (rho_dot, theta_dot)


def _cylindrical(
    points: jt.Float[jax.Array, "n_point 3"],
) -> tuple[
    jt.Float[jax.Array, " n_point"],
    jt.Float[jax.Array, " n_point"],
    jt.Float[jax.Array, " n_point"],
]:
    """`(R, Z, zeta)` of Cartesian points; `zeta` is the cylindrical angle here."""
    return (
        jnp.hypot(points[:, 0], points[:, 1]),
        points[:, 2],
        jnp.arctan2(points[:, 1], points[:, 0]),
    )


def _forward_rz(
    fixed: _FixedInputs,
    rho: jax.Array,
    theta: jax.Array,
    r_lmn: jax.Array,
    z_lmn: jax.Array,
    zeta: jax.Array,
) -> tuple[jt.Float[jax.Array, " n_point"], jt.Float[jax.Array, " n_point"]]:
    """`desc_utils.evaluate_rz` at fixed `(rho, theta)`, with the coefficients and
    `zeta` as positional arguments so `jax.jvp` can differentiate them."""
    return desc_utils.evaluate_rz(
        _geometry(fixed, r_lmn, z_lmn),
        rho,
        theta,
        zeta,
        extend_beyond_boundary=True,
    )


def _geometry(
    fixed: _FixedInputs, r_lmn: jax.Array, z_lmn: jax.Array
) -> desc_types.DescFourierZernikeCoefficients:
    """The fixed geometry carrying the coefficients currently being differentiated."""
    return fixed.coefficients.model_copy(update={"r_lmn": r_lmn, "z_lmn": z_lmn})
