# Laplace approximation of the posterior at the MAP: a Gaussian in the unconstrained
# space whose precision is the exact Hessian of the energy `U(z)`, in one compiled call.

import functools
import time
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import jaxtyping as jt
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_model_utils,
    inference_target_types,
    inference_types,
    likelihood_utils,
    posterior_utils,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    model_storage_utils,
)


class LaplaceApproximation(NamedTuple):
    """The Gaussian `N(mean, covariance)` standing in for the posterior, in `z`."""

    mean: jt.Float[jax.Array, " n_param"]
    """The expansion point: the MAP, in the unconstrained space."""

    covariance: jt.Float[jax.Array, "n_param n_param"]
    """`hessian^-1`; NaN when the Hessian is not positive definite, since no Gaussian
    has it as a precision and no band can be formed."""

    hessian: jt.Float[jax.Array, "n_param n_param"]
    """Exact Hessian of `U(z)` at the mean, symmetrized: the likelihood, the prior and
    the transform log-Jacobians, all by `jax.hessian`."""

    information: jt.Float[jax.Array, "n_param n_param"]
    """`J^T J`, `J` the whitened residual Jacobian: the Fisher information of the data,
    which `information_gain_bits` reads instead of the data-dependent Hessian."""

    prior_precision: jt.Float[jax.Array, "n_param n_param"]
    """Hessian of the negative log-prior in `z`, transform log-Jacobians included."""

    gradient_norm: float
    """`|grad U(z)|` at the mean; small when the MAP it expands around has converged."""

    newton_decrement: float
    """`g^T H^-1 g` at the mean, `H` the Hessian: the squared distance to the local
    quadratic model's minimum in posterior standard deviations, inflated where `H` is
    near singular and NaN where it is not positive definite."""

    curvature_min_eigenvalue: float
    """Smallest eigenvalue of `H`, near zero along a direction the data and the prior
    leave unconstrained, such as a calibration at its box edge, negative at a saddle."""

    curvature_condition_number: float
    """Largest over smallest eigenvalue of `H`."""

    seconds: float
    """Wall-clock time of the fit, compilation included."""


def fit_laplace(
    posterior: posterior_utils.Posterior,
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> LaplaceApproximation:
    """Fit the Gaussian at `posterior.z_init`, the MAP, with the exact Hessian of `U`:
    exact for a linear forward model with Gaussian noise and prior, a local second-order
    approximation otherwise, blind to multimodality and skew.

    Args:
        posterior: The posterior with `z_init` at the mode, as
            `map_optimization_utils.run_map` returns it.
        settings: The run configuration, supplying the forward resolutions.
        priors: Priors on both profiles.
        target: The observed data.
        surrogate: The equilibrium forward model.
        fixed_coefficients: Geometry the MAP held frozen; None solves for it.

    Returns:
        The fitted Gaussian, the matrices it was assembled from, and the curvature
        diagnostics at its mean.

    References:
        Laplace approximation: Murphy 2023, Probabilistic Machine Learning: Advanced
        Topics, sections 7.4.3 and 17.3.2. Exact in the linear-Gaussian case: Attia et
        al. 2018, eqs. (3)-(5), https://doi.org/10.1088/1361-6420/aad210. `J^T J` as
        the Fisher information of the readings linearized at the mode, plus the prior
        precision: Lopez-Fidalgo 2023, Theorem 3.2 and eq. (4.5),
        https://doi.org/10.1007/978-3-031-35918-7. In fusion integrated data analysis:
        Fischer et al. 2024, section 2.7, https://arxiv.org/abs/2411.09270. As the
        design utility: Beck et al. 2018, eq. (26),
        https://doi.org/10.1016/j.cma.2018.01.053.
    """
    started = time.perf_counter()
    z = jnp.asarray(posterior.z_init)
    residuals = functools.partial(
        residuals_of_z,
        posterior=posterior,
        settings=settings,
        target=target,
        surrogate=surrogate,
        flux_guess=inference_model_utils.precompute_flux_guess(
            settings, priors, target, surrogate, fixed_coefficients
        ),
        fixed_coefficients=fixed_coefficients,
    )
    # One compiled call for every derivative: dispatched eagerly, the same matrices
    # cost about twenty seconds, compiled a few.
    curvature = jax.jit(
        functools.partial(
            _curvature, posterior=posterior, priors=priors, residuals=residuals
        )
    )
    hessian, covariance, information, precision, gradient, eigenvalues = curvature(z)
    return LaplaceApproximation(
        mean=z,
        covariance=covariance,
        hessian=hessian,
        information=information,
        prior_precision=precision,
        gradient_norm=float(jnp.linalg.norm(gradient)),
        newton_decrement=float(gradient @ covariance @ gradient),
        curvature_min_eigenvalue=float(eigenvalues[0]),
        curvature_condition_number=float(eigenvalues[-1] / eigenvalues[0]),
        seconds=time.perf_counter() - started,
    )


def sample_laplace(
    approximation: LaplaceApproximation, key: jax.Array, n_samples: int
) -> jt.Float[jax.Array, "n_sample n_param"]:
    """Draw from the fitted Gaussian, `z = mean + L eps` with `L L^T` the covariance.

    Args:
        approximation: The fitted Gaussian.
        key: PRNG key of the draws.
        n_samples: How many to draw.

    Returns:
        One row per draw, in the posterior's `site_names` order.
    """
    standard_normal_draws = jax.random.normal(
        key, (n_samples, approximation.mean.shape[0])
    )
    return (
        approximation.mean
        + standard_normal_draws @ jnp.linalg.cholesky(approximation.covariance).T
    )


def information_gain_bits(
    approximation: LaplaceApproximation,
) -> jt.Float[jax.Array, ""]:
    """Expected information gain of the data under the Gaussian approximation, in bits.

    `0.5 * log2 det(I + P^-1/2 J^T J P^-1/2)`, the expected prior-to-posterior
    Kullback-Leibler divergence of the model linearized at the mean, with Gaussian prior
    precision `P` (Attia et al. 2018, eq. (17); Huan et al. 2024, Acta Numerica,
    eq. (3.33), https://arxiv.org/abs/2407.16212), in bits where both use nats.
    Traceable, so a placement objective can differentiate it.

    Args:
        approximation: The fitted Gaussian.

    Returns:
        The gain, zero for data that constrain nothing beyond the prior.
    """
    root = jnp.linalg.cholesky(approximation.prior_precision)
    half_whitened = jax.scipy.linalg.solve_triangular(
        root, approximation.information, lower=True
    )
    whitened = jax.scipy.linalg.solve_triangular(root, half_whitened.T, lower=True)
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (whitened + whitened.T))
    return 0.5 * jnp.sum(jnp.log2(1.0 + jnp.maximum(eigenvalues, 0.0)))


def negative_log_prior(
    z: jt.Float[NpOrJaxArray, " n_param"],
    posterior: posterior_utils.Posterior,
    priors: priors_utils.ProfilePriors,
) -> jt.Float[jax.Array, ""]:
    """`-log p(z)` of the prior in the unconstrained space, log-Jacobians included.

    Args:
        z: Flat unconstrained parameter vector.
        posterior: The posterior whose transforms define `z`.
        priors: Priors on both profiles.

    Returns:
        The negative log-prior density at `z`.
    """
    values = posterior.constrain(z)
    total = jnp.zeros(())
    for index, name in enumerate(posterior.site_names):
        transform = posterior.transforms[name]
        total = total - priors[name].log_prob(values[name])
        total = total - transform.log_abs_det_jacobian(z[index], values[name])
    return total


def prior_precision(
    z: jt.Float[NpOrJaxArray, " n_param"],
    posterior: posterior_utils.Posterior,
    priors: priors_utils.ProfilePriors,
) -> jt.Float[jax.Array, "n_param n_param"]:
    """Hessian of `negative_log_prior` at `z`; constant for Gaussian priors.

    Args:
        z: Flat unconstrained parameter vector.
        posterior: The posterior whose transforms define `z`.
        priors: Priors on both profiles.

    Returns:
        The prior precision matrix in the unconstrained space.
    """
    return jax.hessian(
        functools.partial(negative_log_prior, posterior=posterior, priors=priors)
    )(jnp.asarray(z))


def residuals_of_z(
    z: jt.Float[NpOrJaxArray, " n_param"],
    posterior: posterior_utils.Posterior,
    settings: inference_types.PlasmaEstimationSettings,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    flux_guess: jt.Float[NpOrJaxArray, "n_point 2"] | None,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None,
) -> jt.Float[jax.Array, " n_residual"]:
    """The whitened residual vector as a function of the flat unconstrained vector.

    Its Jacobian `J` gives the Fisher information `J^T J` of `information_gain_bits`
    and of the placement criterion; `U(z)` is `-log prior + 0.5 |r(z)|^2` plus a
    constant.

    Args:
        z: Flat unconstrained parameter vector.
        posterior: The posterior whose transforms define `z`.
        settings: The run configuration, supplying the forward resolutions.
        target: The observed data.
        surrogate: The equilibrium forward model.
        flux_guess: Precomputed per-point flux seed, or None.
        fixed_coefficients: Geometry to hold frozen; None solves for it.

    Returns:
        `(predicted - observed) / sigma` per channel, zero on masked channels.
    """
    point = posterior.constrain(z)
    profiles = inference_model_utils.profiles_from_point(settings, point)
    return likelihood_utils.residuals(
        profiles,
        inference_model_utils.equilibrium_coefficients(
            surrogate, profiles, fixed_coefficients
        ),
        target,
        maxiter=settings.inverse_mapping_maxiter,
        interferometer_n_sample=settings.interferometer_n_sample,
        flux_guess=flux_guess,
        calibration_corrections=inference_model_utils.calibration_corrections(
            point, target.thomson_signals
        ),
    ).residual


def _curvature(
    z: jt.Float[jax.Array, " n_param"],
    posterior: posterior_utils.Posterior,
    priors: priors_utils.ProfilePriors,
    residuals: Callable[
        [jt.Float[jax.Array, " n_param"]], jt.Float[jax.Array, " n_residual"]
    ],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """The exact Hessian of `U` at `z`, its inverse where it is positive definite, the
    Fisher information `J^T J`, the prior precision, the gradient and the
    eigenvalues."""
    hessian = jax.hessian(posterior.potential)(z)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = jnp.linalg.eigvalsh(hessian)
    inverse = jnp.linalg.inv(hessian)
    covariance = jax.lax.select(
        eigenvalues[0] > 0.0,
        0.5 * (inverse + inverse.T),
        jnp.full_like(inverse, jnp.nan),
    )
    jacobian = jax.jacfwd(residuals)(z)
    return (
        hessian,
        covariance,
        jacobian.T @ jacobian,
        prior_precision(z, posterior, priors),
        jax.grad(posterior.potential)(z),
        eigenvalues,
    )
