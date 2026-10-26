import functools

import jax
import jax.numpy as jnp
import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.inverse_mapping import (
    inverse_map_utils,
)


def _toy_coefficients() -> desc_types.DescFourierZernikeCoefficients:
    # Circular cross-section, nfp=1: R = R0 + rho*a*cos(theta), Z = rho*a*sin(theta).
    return desc_types.DescFourierZernikeCoefficients(
        r_lmn=jnp.array([3.0, 1.0]),
        z_lmn=jnp.array([1.0]),
        lambda_lmn=jnp.array([0.0]),
        r_modes=jnp.array([[0, 0, 0], [1, 1, 0]]),
        z_modes=jnp.array([[1, -1, 0]]),
        lambda_modes=jnp.array([[1, -1, 0]]),
        n_field_periods=1,
    )


def _points_from_flux(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rho: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    r, z = desc_utils.evaluate_rz(coefficients, rho, theta, phi)
    return jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)


def test_implicit_flux_solve_recovers_flux_coordinates() -> None:
    coefficients = _toy_coefficients()
    rho = jnp.array([0.3, 0.7])
    theta = jnp.array([0.5, 2.5])
    phi = jnp.zeros_like(rho)
    points = _points_from_flux(coefficients, rho, theta, phi)

    solved_rho, solved_theta = inverse_map_utils.implicit_flux_solve(
        coefficients, points
    )
    np.testing.assert_allclose(np.asarray(solved_rho), np.asarray(rho), atol=1e-10)
    np.testing.assert_allclose(np.asarray(solved_theta), np.asarray(theta), atol=1e-10)


def test_implicit_gradient_matches_unrolled() -> None:
    """The implicit-diff VJP equals backprop through the unrolled Newton scan.

    At a converged root the unrolled-scan gradient converges to the implicit-function-
    theorem gradient, so on interior points the two must agree to solver precision.
    """
    coefficients = _toy_coefficients()
    rho = jnp.array([0.3, 0.6, 0.9])
    theta = jnp.array([0.5, 2.0, 4.0])
    phi = jnp.zeros_like(rho)
    points = _points_from_flux(coefficients, rho, theta, phi)

    def rho_sum_unrolled(r_lmn: jax.Array) -> jax.Array:
        local = coefficients.model_copy(update={"r_lmn": r_lmn})
        return jnp.sum(desc_utils.xyz_to_flux_coordinates(local, points).rho)

    def rho_sum_implicit(r_lmn: jax.Array) -> jax.Array:
        local = coefficients.model_copy(update={"r_lmn": r_lmn})
        solved_rho, _ = inverse_map_utils.implicit_flux_solve(local, points)
        return jnp.sum(solved_rho)

    r_lmn = jnp.asarray(coefficients.r_lmn)
    gradient_unrolled = np.asarray(jax.grad(rho_sum_unrolled)(r_lmn))
    gradient_implicit = np.asarray(jax.grad(rho_sum_implicit)(r_lmn))
    np.testing.assert_allclose(gradient_implicit, gradient_unrolled, rtol=1e-6)


def test_implicit_flux_solve_with_precomputed_guess() -> None:
    """A precomputed seed gives the same solution as the built-in grid search."""
    coefficients = _toy_coefficients()
    rho = jnp.array([0.2, 0.5, 0.8])
    theta = jnp.array([0.3, 1.5, 5.0])
    phi = jnp.zeros_like(rho)
    points = _points_from_flux(coefficients, rho, theta, phi)

    guess = jnp.stack(desc_utils.initial_guess(coefficients, points), axis=1)
    rho_default, _ = inverse_map_utils.implicit_flux_solve(coefficients, points)
    rho_seeded, _ = inverse_map_utils.implicit_flux_solve(
        coefficients, points, guess=guess
    )
    np.testing.assert_allclose(
        np.asarray(rho_seeded), np.asarray(rho_default), atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(rho_seeded), np.asarray(rho), atol=1e-10)


def _solved_rho(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    r_lmn: jax.Array,
    z_lmn: jax.Array,
    points: jax.Array,
) -> jax.Array:
    """`rho` of the implicit solve at the given coefficients and points."""
    local = coefficients.model_copy(update={"r_lmn": r_lmn, "z_lmn": z_lmn})
    return inverse_map_utils.implicit_flux_solve(local, points)[0]


def _rho_sum(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    points: jax.Array,
    r_lmn: jax.Array,
) -> jax.Array:
    """A scalar of the solve, `sum(rho^2)`, as a function of `r_lmn` alone."""
    return jnp.sum(
        jnp.square(
            _solved_rho(coefficients, r_lmn, jnp.asarray(coefficients.z_lmn), points)
        )
    )


def _interior_points(
    coefficients: desc_types.DescFourierZernikeCoefficients,
) -> jnp.ndarray:
    rho = jnp.array([0.3, 0.6, 0.9])
    theta = jnp.array([0.5, 2.0, 4.0])
    return _points_from_flux(coefficients, rho, theta, jnp.zeros_like(rho))


def test_rho_jvp_matches_central_differences() -> None:
    """The implicit JVP of `rho` equals central differences of the converged solve, in a
    joint direction of the coefficients and the points."""
    coefficients = _toy_coefficients()
    points = _interior_points(coefficients)
    primals = (jnp.asarray(coefficients.r_lmn), jnp.asarray(coefficients.z_lmn), points)
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    tangents = tuple(
        jax.random.normal(key, primal.shape)
        for key, primal in zip(keys, primals, strict=True)
    )
    rho_of = functools.partial(_solved_rho, coefficients)

    _, jvp = jax.jvp(rho_of, primals, tangents)
    step = 1e-5
    shifted_up = [p + step * t for p, t in zip(primals, tangents, strict=True)]
    shifted_down = [p - step * t for p, t in zip(primals, tangents, strict=True)]
    finite_difference = (rho_of(*shifted_up) - rho_of(*shifted_down)) / (2 * step)
    np.testing.assert_allclose(
        np.asarray(jvp), np.asarray(finite_difference), rtol=1e-6
    )


def test_rho_forward_and_reverse_jacobians_agree() -> None:
    """`jacfwd` and `jacrev` of `rho` with respect to `r_lmn` give the same matrix,
    since reverse mode is the transpose of the JVP rule."""
    coefficients = _toy_coefficients()
    points = _interior_points(coefficients)
    rho_of_r = functools.partial(
        _solved_rho, coefficients, z_lmn=jnp.asarray(coefficients.z_lmn), points=points
    )
    r_lmn = jnp.asarray(coefficients.r_lmn)

    jacobian_forward = np.asarray(jax.jacfwd(rho_of_r)(r_lmn))
    jacobian_reverse = np.asarray(jax.jacrev(rho_of_r)(r_lmn))
    assert jacobian_forward.shape == (points.shape[0], r_lmn.shape[0])
    np.testing.assert_allclose(jacobian_forward, jacobian_reverse, rtol=1e-12)


def test_rho_hessian_matches_central_differences_of_gradient() -> None:
    """`jax.hessian` of a scalar of `rho` with respect to the coefficients is symmetric
    and equals central differences of `jax.grad`."""
    coefficients = _toy_coefficients()
    points = _interior_points(coefficients)
    scalar = functools.partial(_rho_sum, coefficients, points)
    r_lmn = jnp.asarray(coefficients.r_lmn)

    hessian = np.asarray(jax.hessian(scalar)(r_lmn))
    gradient = jax.grad(scalar)
    step = 1e-5
    columns = []
    for index in range(r_lmn.shape[0]):
        direction = jnp.zeros_like(r_lmn).at[index].set(step)
        columns.append(
            (gradient(r_lmn + direction) - gradient(r_lmn - direction)) / (2 * step)
        )
    finite_difference = np.stack([np.asarray(column) for column in columns], axis=1)
    np.testing.assert_allclose(hessian, hessian.T, rtol=1e-12)
    np.testing.assert_allclose(hessian, finite_difference, rtol=1e-6)
