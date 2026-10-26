import desc.basis
import desc.equilibrium
import numpy as np
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
    testing_utils,
)

_INVERSE_MAP_TOL = 1e-12
_INVERSE_MAP_MAX_ITERATIONS = 100


@pytest.fixture(scope="module")
def shaped_coefficients() -> desc_types.DescFourierZernikeCoefficients:
    return testing_utils.coefficients_from_equilibrium(
        testing_utils.rotating_ellipse_equilibrium()
    )


def test_xyz_to_flux_coordinates_matches_map_coordinates() -> None:
    """The inverse map agrees with DESC's `map_coordinates` to 1e-12.

    Builds an `Equilibrium` as ground truth, extracts its Fourier-Zernike coefficients,
    and checks that `xyz_to_flux_coordinates` recovers the same `(rho, theta, zeta)` as
    `Equilibrium.map_coordinates` for a batch of interior points.
    """
    equilibrium = desc.equilibrium.Equilibrium(L=8, M=6, N=2, NFP=5, sym=True)
    coefficients = testing_utils.coefficients_from_equilibrium(equilibrium)

    # Interior points from the forward map (guaranteed inside the last closed surface).
    rng = np.random.default_rng(0)
    n = 40
    rho = rng.uniform(0.2, 0.85, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    zeta = rng.uniform(0.0, 2.0 * np.pi, n)
    r, z = desc_utils.evaluate_rz(coefficients, rho, theta, zeta)
    r, z = np.asarray(r), np.asarray(z)
    points = np.column_stack([r * np.cos(zeta), r * np.sin(zeta), z])

    flux = desc_utils.xyz_to_flux_coordinates(
        coefficients, points, tol=1e-14, maxiter=100
    )

    coords = np.column_stack(
        [
            np.hypot(points[:, 0], points[:, 1]),
            np.arctan2(points[:, 1], points[:, 0]),
            points[:, 2],
        ]
    )
    mapped = np.asarray(
        equilibrium.map_coordinates(
            coords,
            inbasis=("R", "phi", "Z"),
            outbasis=("rho", "theta", "zeta"),
            period=(np.inf, 2.0 * np.pi, np.inf),
            tol=1e-14,
            maxiter=100,
        )
    )

    def _wrapped(delta: np.ndarray) -> np.ndarray:
        return (delta + np.pi) % (2.0 * np.pi) - np.pi

    np.testing.assert_allclose(np.asarray(flux.rho), mapped[:, 0], atol=1e-12)
    np.testing.assert_allclose(
        _wrapped(np.asarray(flux.theta) - mapped[:, 1]), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        _wrapped(np.asarray(flux.zeta) - mapped[:, 2]), 0.0, atol=1e-12
    )


def _desc_basis_series(
    coefficients: NpOrJaxArray,
    modes: NpOrJaxArray,
    n_field_periods: int,
    rho: NpOrJaxArray,
    theta: NpOrJaxArray,
    zeta: NpOrJaxArray,
    d_rho: int = 0,
    d_theta: int = 0,
) -> np.ndarray:
    """A Fourier-Zernike series evaluated mode by mode with DESC's own basis functions.

    The independent reference for `evaluate_rz` / `evaluate_rz_jacobian`, which contract
    the toroidal factor into `(l, m)` amplitudes once per call and evaluate the radial
    factor as a static polynomial matmul instead of calling
    `desc.basis.zernike_radial` per step. Both must agree to round-off.
    """
    modes = np.asarray(modes)
    radial = desc.basis.zernike_radial(
        np.asarray(rho)[:, None], modes[:, 0], modes[:, 1], dr=d_rho
    )
    poloidal = desc.basis.fourier(np.asarray(theta)[:, None], modes[:, 1], dt=d_theta)
    toroidal = desc.basis.fourier(
        np.asarray(zeta)[:, None], modes[:, 2], NFP=n_field_periods
    )
    return np.asarray(radial * poloidal * toroidal) @ np.asarray(coefficients)


def test_evaluate_rz_and_jacobian_match_desc_basis_series(
    shaped_coefficients: desc_types.DescFourierZernikeCoefficients,
) -> None:
    """`(R, Z)` and all four partials match the mode-by-mode `desc.basis` series."""
    coefficients = shaped_coefficients
    n = 24
    rng = np.random.default_rng(23)
    rho = rng.uniform(0.0, 1.0, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    zeta = rng.uniform(0.0, 2.0 * np.pi, n)
    nfp = coefficients.n_field_periods

    r, z = desc_utils.evaluate_rz(coefficients, rho, theta, zeta)
    jacobian = desc_utils.evaluate_rz_jacobian(coefficients, rho, theta, zeta)
    for name, value, args in (
        ("R", r, (coefficients.r_lmn, coefficients.r_modes, {})),
        ("Z", z, (coefficients.z_lmn, coefficients.z_modes, {})),
        (
            "dR/drho",
            jacobian.dr_drho,
            (coefficients.r_lmn, coefficients.r_modes, {"d_rho": 1}),
        ),
        (
            "dR/dtheta",
            jacobian.dr_dtheta,
            (coefficients.r_lmn, coefficients.r_modes, {"d_theta": 1}),
        ),
        (
            "dZ/drho",
            jacobian.dz_drho,
            (coefficients.z_lmn, coefficients.z_modes, {"d_rho": 1}),
        ),
        (
            "dZ/dtheta",
            jacobian.dz_dtheta,
            (coefficients.z_lmn, coefficients.z_modes, {"d_theta": 1}),
        ),
    ):
        lmn, modes, kwargs = args
        reference = _desc_basis_series(lmn, modes, nfp, rho, theta, zeta, **kwargs)
        np.testing.assert_allclose(
            np.asarray(value), reference, rtol=1e-12, atol=1e-12, err_msg=name
        )


@pytest.mark.parametrize("extend_beyond_boundary", [False, True])
def test_evaluate_rz_jacobian_matches_finite_difference(
    shaped_coefficients: desc_types.DescFourierZernikeCoefficients,
    extend_beyond_boundary: bool,
) -> None:
    """The analytic `(rho, theta)` partials match central differences of `evaluate_rz`.

    Points straddling `rho = 1` are dropped: the continuation makes `d/drho` genuinely
    discontinuous there, so a difference across the kink is not a derivative.
    """
    coefficients = shaped_coefficients
    n = 24
    rng = np.random.default_rng(29)
    high = 1.8 if extend_beyond_boundary else 0.99
    rho = rng.uniform(0.15, high, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    zeta = rng.uniform(0.0, 2.0 * np.pi, n)

    step = 1e-6
    jacobian = desc_utils.evaluate_rz_jacobian(
        coefficients, rho, theta, zeta, extend_beyond_boundary
    )
    keep = np.abs(rho - 1.0) > 10.0 * step
    for variable, expected in (
        ("rho", (jacobian.dr_drho, jacobian.dz_drho)),
        ("theta", (jacobian.dr_dtheta, jacobian.dz_dtheta)),
    ):
        rho_step = step if variable == "rho" else 0.0
        theta_step = step if variable == "theta" else 0.0
        plus = desc_utils.evaluate_rz(
            coefficients,
            rho + rho_step,
            theta + theta_step,
            zeta,
            extend_beyond_boundary,
        )
        minus = desc_utils.evaluate_rz(
            coefficients,
            rho - rho_step,
            theta - theta_step,
            zeta,
            extend_beyond_boundary,
        )
        for index, name in enumerate(("R", "Z")):
            difference = (np.asarray(plus[index]) - np.asarray(minus[index])) / (
                2.0 * step
            )
            np.testing.assert_allclose(
                np.asarray(expected[index])[keep],
                difference[keep],
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"d{name}/d{variable}",
            )


def test_fourier_zernike_mode_tables_without_stellarator_symmetry() -> None:
    """Dropping the symmetry gives all three bases the same, larger mode table."""
    symmetric = desc_types.DescResolution(
        radial_resolution=4,
        poloidal_resolution=3,
        toroidal_resolution=2,
        n_field_periods=5,
    )
    asymmetric = symmetric.model_copy(update={"is_stellarator_symmetric": False})

    r_symmetric, z_symmetric, _ = desc_utils.fourier_zernike_mode_tables(symmetric)
    r_modes, z_modes, lambda_modes = desc_utils.fourier_zernike_mode_tables(asymmetric)

    np.testing.assert_array_equal(r_modes, z_modes)
    np.testing.assert_array_equal(r_modes, lambda_modes)
    assert len(r_modes) == len(r_symmetric) + len(z_symmetric)
