# One inference run, from a target to a record ready to store: the MAP from every start,
# then the 5-95 band of the best converged start's Laplace Gaussian, or NUTS's.

import functools
import secrets
import time
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    inference_model_utils,
    inference_target_types,
    inference_types,
    laplace_approximation_utils,
    likelihood_utils,
    map_optimization_utils,
    mcmc_utils,
    posterior_utils,
    priors_utils,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    model_storage_utils,
)

DEFAULT_N_PROFILE_POINTS = 120
"""Radial points every reported profile and band is evaluated on."""

DEFAULT_N_POSTERIOR_SAMPLES = 512
"""Draws taken from a Gaussian posterior to turn it into a profile band."""

_BAND_PERCENTILES = (5.0, 95.0)
"""The interval every method reports, so any two bands overlay in one figure."""

DISTINCT_MODE_TOLERANCE = 1.0
"""Gap in `U`, in nats, above the best converged start beyond which another converged
start counts as a distinct mode, reported on the record rather than blended in."""


class PointMisfit(NamedTuple):
    """Goodness of fit of one point estimate against the data it was fitted to."""

    chi_squared: float
    """`sum(residual^2)` over the unmasked channels."""

    n_degrees_of_freedom: int
    """Unmasked channels minus fitted parameters."""


class StartSelection(NamedTuple):
    """Which MAP starts a run reads, decided once for both recipes."""

    converged: jt.Bool[np.ndarray, " n_start"]
    """Per start, whether it stopped before `MapSettings.max_iterations` at a positive
    definite curvature with a Newton decrement below `decrement_tolerance` and a reduced
    chi-squared below `reduced_chi_squared_tolerance`."""

    candidates: jt.Bool[np.ndarray, " n_start"]
    """The converged starts; with none converged, those with a positive-definite
    curvature, or every start. The best is chosen among them and the per-start band
    widths, gains and modes are read."""

    best: int
    """The candidate with the lowest `U`, whose point and Gaussian the record reports
    and NUTS starts from."""

    n_modes: int
    """Candidates ending more than `DISTINCT_MODE_TOLERANCE` above the best in `U`."""

    mode_gap: float
    """Gap in `U` from the best to the nearest of those; zero when there is none."""


class BandOutcome(NamedTuple):
    """What one band method produced around the MAP, before it goes into a record."""

    band: inference_types.PlasmaProfilePosterior
    """The reported profile band."""

    arrays: dict[str, np.ndarray]
    """Arrays a figure needs: the fitted Gaussian, its draws, any history."""

    scalars: dict[str, Any]
    """Timings and diagnostics, by record field name."""


def point_misfit(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    point: Mapping[str, jt.Float[NpOrJaxArray, ""]],
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> PointMisfit:
    """Whitened chi-squared of one per-site point, scored with the calibration
    corrections it carries, and the degrees of freedom for it; near one per degree of
    freedom the data is explained within its uncertainties.

    Args:
        settings: The run configuration, supplying the forward resolutions.
        priors: Priors on both profiles, whose count fixes the fitted parameters.
        target: The observed data.
        surrogate: The equilibrium forward model.
        point: The point estimate to score, one value per sample site.
        fixed_coefficients: Geometry the run held frozen; None solves for it. Must
            match what the run fitted under, or the misfit reported on the record is
            not the misfit the likelihood saw.

    Returns:
        The chi-squared and the degrees of freedom.
    """
    profiles = inference_model_utils.profiles_from_point(settings, point)
    whitened = likelihood_utils.residuals(
        profiles,
        inference_model_utils.equilibrium_coefficients(
            surrogate, profiles, fixed_coefficients
        ),
        target,
        maxiter=settings.inverse_mapping_maxiter,
        interferometer_n_sample=settings.interferometer_n_sample,
        flux_guess=inference_model_utils.precompute_flux_guess(
            settings, priors, target, surrogate, fixed_coefficients
        ),
        calibration_corrections=inference_model_utils.calibration_corrections(
            point, target.thomson_signals
        ),
    )
    return PointMisfit(
        chi_squared=float(jnp.sum(jnp.square(whitened.residual))),
        n_degrees_of_freedom=int(jnp.sum(whitened.valid)) - len(priors),
    )


def run_inference(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    truth: desc_truth_types.DescTruth,
    design_point: diagnostic_design_point.DiagnosticDesignPoint | None = None,
    n_profile_points: int = DEFAULT_N_PROFILE_POINTS,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
    warm_start: Mapping[str, float] | None = None,
) -> tuple[inference_types.PlasmaStateInferenceResult, dict[str, np.ndarray]]:
    """Run one inference and assemble its record and arrays: the MAP from every start,
    the curvature at each, then the band, `map_laplace` the Gaussian at the best
    converged start alone and `mcmc_nuts` sampling from that start.

    Args:
        settings: The run configuration, including which method to run.
        priors: Priors on both profiles.
        target: The observed data.
        surrogate: The equilibrium forward model.
        truth: The ground truth the target was generated from.
        design_point: The instrument the target was built on, stored on the record so a
            run says what noise model produced it; the pinned design point when None.
        n_profile_points: Radial points the reported profiles use.
        fixed_coefficients: Geometry to hold frozen for every parameter value, which
            makes this the conventional inference that assumes an equilibrium; None
            predicts it from the pressure, which is what every other study here does.
            See `inference_model_utils.plasma_model`.
        warm_start: A point to start from once every prior draw has failed, or first
            when `MapSettings.warm_start_first` is set, one value per sample site in
            the constrained space; None tries the prior draws only.

    Returns:
        The record and the arrays a figure needs, ready for `save_inference`.
    """
    if truth.machine_scale != surrogate.machine_scale:
        raise ValueError(
            f"the truth is at {truth.machine_scale} but the surrogate at "
            f"{surrogate.machine_scale}; read both at one machine scale."
        )
    started = time.perf_counter()
    rho = jnp.linspace(0.0, 1.0, n_profile_points)
    map_results: list[map_optimization_utils.MapResult] = []
    approximations: list[laplace_approximation_utils.LaplaceApproximation] = []
    misfits: list[PointMisfit] = []
    map_starts = inference_model_utils.start_points(settings, priors)
    warm_index = -1
    if warm_start is not None:
        if set(warm_start) != set(priors):
            raise ValueError(
                f"the warm start has sites {sorted(warm_start)} but the priors "
                f"{sorted(priors)}."
            )
        warm_index = 0 if settings.map.warm_start_first else len(map_starts)
        map_starts.insert(
            warm_index,
            {
                name: jnp.asarray(value, dtype=float)
                for name, value in warm_start.items()
            },
        )
    # Every requested start runs; past them the next start, a new prior draw and last
    # the warm start unless it came first, is tried only while none has converged, so a
    # failed start is retried rather than outvoted.
    for result in map_optimization_utils.run_map(
        settings, priors, target, surrogate, map_starts, fixed_coefficients
    ):
        map_results.append(result)
        # The curvature at every start's end point, whatever the recipe: it decides
        # convergence, its diagnostics and band width go on the record, and
        # `map_laplace` samples the best's.
        approximations.append(
            laplace_approximation_utils.fit_laplace(
                result.posterior,
                settings,
                priors,
                target,
                surrogate,
                fixed_coefficients,
            )
        )
        misfits.append(
            point_misfit(
                settings, priors, target, surrogate, result.point, fixed_coefficients
            )
        )
        converged = _converged(map_results, approximations, misfits, settings)
        if len(map_results) >= settings.map.n_starts and converged.any():
            break
        if settings.verbose and settings.map.n_starts <= len(map_results) < len(
            map_starts
        ):
            print(
                f"[MAP] start {len(map_results) - 1} not converged (Newton decrement "
                f"{float(approximations[-1].newton_decrement):.3g}, chi^2 / dof "
                f"{_reduced_chi_squared(misfits[-1]):.3g}), trying the next start"
            )
    starts = _select_starts(map_results, approximations, misfits, settings)
    map_result = map_results[starts.best]
    width_ne, width_te, gains = _start_bands(
        settings, priors, map_result.posterior, approximations, starts.candidates, rho
    )
    initial_profiles = inference_model_utils.profiles_from_point(
        settings, map_result.start_point
    )
    map_profiles = inference_model_utils.profiles_from_point(settings, map_result.point)
    if settings.method == "map_laplace":
        outcome = _laplace_band(
            settings,
            priors,
            map_result.posterior,
            approximations[starts.best],
            gains[starts.best],
            rho,
        )
    else:
        outcome = _mcmc_band(
            settings,
            priors,
            target,
            surrogate,
            map_result,
            approximations[starts.best],
            rho,
            fixed_coefficients,
        )
    misfit = misfits[starts.best]

    site_names = map_result.posterior.site_names
    start_losses = np.asarray([result.final_loss for result in map_results])
    n_iterations = max(result.n_iterations for result in map_results)
    arrays: dict[str, np.ndarray] = {
        "rho": np.asarray(rho),
        # A start that stopped early is held at its end point for the remaining
        # iterations.
        "start_trajectories": np.stack(
            [
                np.pad(
                    np.stack(
                        [np.asarray(result.trajectory[name]) for name in site_names],
                        axis=1,
                    ),
                    ((0, n_iterations - result.n_iterations), (0, 0)),
                    mode="edge",
                )
                for result in map_results
            ]
        ),
        "trajectory_site_names": np.asarray(site_names),
        "start_log_posterior": -start_losses,
        "start_gradient_norm": np.asarray(
            [result.gradient_norm for result in map_results]
        ),
        "start_gradient_inf_norm": np.asarray(
            [result.gradient_inf_norm for result in map_results]
        ),
        "start_n_iterations": np.asarray(
            [result.n_iterations for result in map_results]
        ),
        "start_stop_reason": np.asarray([result.stop_reason for result in map_results]),
        "start_newton_decrement": np.asarray(
            [approximation.newton_decrement for approximation in approximations]
        ),
        "start_curvature_min_eigenvalue": np.asarray(
            [approximation.curvature_min_eigenvalue for approximation in approximations]
        ),
        "start_curvature_condition_number": np.asarray(
            [
                approximation.curvature_condition_number
                for approximation in approximations
            ]
        ),
        "start_reduced_chi_squared": np.asarray(
            [_reduced_chi_squared(start_misfit) for start_misfit in misfits]
        ),
        "start_converged": starts.converged,
        "start_is_warm": np.arange(len(map_results)) == warm_index,
        "start_band_width_ne": width_ne,
        "start_band_width_te": width_te,
        "start_information_gain_bits": gains,
    }
    for prefix, profiles in (
        ("initial", initial_profiles),
        ("map", map_profiles),
        ("true", truth.profiles),
    ):
        _store_profiles(arrays, prefix, profiles, rho)
    arrays.update(outcome.arrays)

    thomson_sigma = (
        np.asarray(target.thomson_signals.uncertainties)
        if target.thomson_signals is not None
        else np.zeros((0,))
    )
    result = inference_types.PlasmaStateInferenceResult(
        inference_id=secrets.token_hex(4),
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        truth_name=truth.name,
        surrogate_name=surrogate.name,
        machine_scale=surrogate.machine_scale,
        design_point=design_point or diagnostic_design_point.DiagnosticDesignPoint(),
        likelihood_quantities=_likelihood_quantities(target),
        settings=settings,
        map_profiles=map_profiles,
        initial_profiles=posterior_from_profiles(initial_profiles, rho),
        inferred_profiles=outcome.band,
        true_profiles=posterior_from_profiles(truth.profiles, rho),
        log_posterior_map=-map_result.final_loss,
        map_log_posterior_spread=float(np.ptp(start_losses)),
        map_gradient_norm=map_result.gradient_norm,
        map_gradient_inf_norm=map_result.gradient_inf_norm,
        map_newton_decrement=approximations[starts.best].newton_decrement,
        map_converged=bool(starts.converged.any()),
        n_map_starts_converged=int(starts.converged.sum()),
        n_map_modes=starts.n_modes,
        map_mode_gap=starts.mode_gap,
        map_band_width_ratio_ne=float(np.nanmax(width_ne) / width_ne[starts.best]),
        map_band_width_ratio_te=float(np.nanmax(width_te) / width_te[starts.best]),
        chi_squared=misfit.chi_squared,
        n_degrees_of_freedom=misfit.n_degrees_of_freedom,
        map_iterations=map_result.n_iterations,
        n_thomson_channels=int(thomson_sigma.size),
        n_thomson_channels_masked=int((thomson_sigma <= 0.0).sum()),
        map_time_seconds=sum(result.seconds for result in map_results),
        laplace_time_seconds=sum(
            approximation.seconds for approximation in approximations
        ),
        total_time_seconds=time.perf_counter() - started,
        **outcome.scalars,
    )
    return result, arrays


def profile_samples_in_si(
    settings: inference_types.PlasmaEstimationSettings,
    name: str,
    samples: jt.Float[NpOrJaxArray, "n_sample n_param"],
    rho: jt.Float[jax.Array, " n_rho"],
) -> np.ndarray:
    """One profile's sample curves in SI units, on the reporting radial grid.

    Args:
        settings: The run configuration, supplying the family and its grid.
        name: `electron_density` or `electron_temperature`.
        samples: Parameter draws of that profile, one row per draw.
        rho: Radial grid to evaluate on.

    Returns:
        One curve per draw.
    """
    unit = (
        profile_parametrization_types.DENSITY_UNIT
        if name == "electron_density"
        else profile_parametrization_types.TEMPERATURE_UNIT
    )
    return np.asarray(
        profile_parametrization_types.to_si(
            settings.profile_family,
            _evaluate_samples(settings, samples, rho),
            unit,
        )
    )


def band_from_unconstrained_draws(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    posterior: posterior_utils.Posterior,
    draws: jt.Float[NpOrJaxArray, "n_sample n_param"],
    rho: jt.Float[jax.Array, " n_rho"],
) -> tuple[inference_types.PlasmaProfilePosterior, np.ndarray, np.ndarray]:
    """The reported band of unconstrained draws, plus the parameter draws behind it.

    The one route from a set of draws in `z` to a profile band, shared by the Laplace
    approximation and the sampler, so their bands differ in the draws alone.

    Args:
        settings: The run configuration, supplying the family and its grid.
        priors: Priors on both profiles, which name the sites.
        posterior: The posterior whose transforms map the draws back.
        draws: Unconstrained draws, one row each, in `posterior.site_names` order.
        rho: Radial grid the band is evaluated on.

    Returns:
        The 5th-to-95th band, and the constrained `ne` and `Te` parameter draws.
    """
    density_samples, temperature_samples = _constrained_samples(
        posterior, jnp.asarray(draws), priors
    )
    return (
        _posterior_from_samples(settings, density_samples, temperature_samples, rho),
        density_samples,
        temperature_samples,
    )


def posterior_from_profiles(
    profiles: profile_parametrization_types.PlasmaProfiles,
    rho: jt.Float[jax.Array, " n_rho"],
) -> inference_types.PlasmaProfilePosterior:
    """One profile set as a zero-width band, as a truth or a start point is."""
    density = np.asarray(profile_parametrization_types.electron_density(profiles, rho))
    temperature = np.asarray(
        profile_parametrization_types.electron_temperature(profiles, rho)
    )
    pressure = np.asarray(profile_parametrization_types.pressure_pascal(profiles, rho))
    return inference_types.PlasmaProfilePosterior(
        rho=np.asarray(rho),
        electron_density_median=density,
        electron_density_lower=density,
        electron_density_upper=density,
        electron_temperature_median=temperature,
        electron_temperature_lower=temperature,
        electron_temperature_upper=temperature,
        pressure_median=pressure,
        pressure_lower=pressure,
        pressure_upper=pressure,
        electron_density_mean=density,
        electron_temperature_mean=temperature,
        pressure_mean=pressure,
    )


def _converged(
    map_results: list[map_optimization_utils.MapResult],
    approximations: list[laplace_approximation_utils.LaplaceApproximation],
    misfits: list[PointMisfit],
    settings: inference_types.PlasmaEstimationSettings,
) -> jt.Bool[np.ndarray, " n_start"]:
    """Per start, whether it stopped before the cap at a positive-definite curvature
    with a Newton decrement below `MapSettings.decrement_tolerance` and, where the data
    leave degrees of freedom, a reduced chi-squared below its tolerance."""
    return np.asarray(
        [
            result.stop_reason != "iteration_cap"
            and float(approximation.curvature_min_eigenvalue) > 0.0
            and float(approximation.newton_decrement) < settings.map.decrement_tolerance
            and (
                misfit.n_degrees_of_freedom <= 0
                or _reduced_chi_squared(misfit)
                < settings.map.reduced_chi_squared_tolerance
            )
            for result, approximation, misfit in zip(
                map_results, approximations, misfits, strict=True
            )
        ],
        dtype=bool,
    )


def _reduced_chi_squared(misfit: PointMisfit) -> float:
    """`chi^2 / dof`, nan for a target with no degrees of freedom."""
    if misfit.n_degrees_of_freedom <= 0:
        return float("nan")
    return misfit.chi_squared / misfit.n_degrees_of_freedom


def _select_starts(
    map_results: list[map_optimization_utils.MapResult],
    approximations: list[laplace_approximation_utils.LaplaceApproximation],
    misfits: list[PointMisfit],
    settings: inference_types.PlasmaEstimationSettings,
) -> StartSelection:
    """Certify the converged starts, pick the best among them (with none converged,
    among the positive-definite starts or else all, so the failure is stored, not
    raised) and count the distinct modes the other candidates ended in."""
    losses = np.asarray([result.final_loss for result in map_results])
    converged = _converged(map_results, approximations, misfits, settings)
    # An unconverged start with a positive-definite curvature still has a Laplace band,
    # so it is reported in place of a lower one without.
    positive_definite = np.asarray(
        [
            float(approximation.curvature_min_eigenvalue) > 0.0
            for approximation in approximations
        ]
    )
    if converged.any():
        candidates = converged
    elif positive_definite.any():
        candidates = positive_definite
    else:
        candidates = np.ones_like(converged)
    best = int(np.flatnonzero(candidates)[np.argmin(losses[candidates])])
    distinct = candidates & (losses > losses[best] + DISTINCT_MODE_TOLERANCE)
    return StartSelection(
        converged=converged,
        candidates=candidates,
        best=best,
        n_modes=int(distinct.sum()),
        mode_gap=float(losses[distinct].min() - losses[best])
        if distinct.any()
        else 0.0,
    )


def _start_bands(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    posterior: posterior_utils.Posterior,
    approximations: list[laplace_approximation_utils.LaplaceApproximation],
    candidates: jt.Bool[np.ndarray, " n_start"],
    rho: jt.Float[jax.Array, " n_rho"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per candidate start, from its own Gaussian: the integral over rho of the
    posterior standard deviation of `ne` and of `Te`, and its information gain in bits;
    nan for every other start."""
    widths = {
        name: np.full(len(approximations), np.nan)
        for name in ("electron_density", "electron_temperature")
    }
    gains = np.full(len(approximations), np.nan)
    key = jax.random.PRNGKey(settings.seed + 3)
    for index in np.flatnonzero(candidates):
        approximation = approximations[index]
        draws = laplace_approximation_utils.sample_laplace(
            approximation,
            jax.random.fold_in(key, int(index)),
            settings.n_posterior_samples,
        )
        for name, samples in zip(
            widths, _constrained_samples(posterior, draws, priors)
        ):
            curves = profile_samples_in_si(settings, name, samples, rho)
            widths[name][index] = np.trapezoid(np.std(curves, axis=0), np.asarray(rho))
        gains[index] = float(
            laplace_approximation_utils.information_gain_bits(approximation)
        )
    return widths["electron_density"], widths["electron_temperature"], gains


def _laplace_band(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    posterior: posterior_utils.Posterior,
    approximation: laplace_approximation_utils.LaplaceApproximation,
    information_gain: float,
    rho: jt.Float[jax.Array, " n_rho"],
) -> BandOutcome:
    """The best start's Gaussian sampled, and the percentile band of its draws; the
    Gaussian itself is stored for the figures that read it back."""
    band, density_samples, temperature_samples = band_from_unconstrained_draws(
        settings,
        priors,
        posterior,
        laplace_approximation_utils.sample_laplace(
            approximation,
            jax.random.PRNGKey(settings.seed + 2),
            settings.n_posterior_samples,
        ),
        rho,
    )
    return BandOutcome(
        band=band,
        arrays={
            "laplace_mean": np.asarray(approximation.mean),
            "laplace_covariance": np.asarray(approximation.covariance),
            "laplace_hessian": np.asarray(approximation.hessian),
            "laplace_information": np.asarray(approximation.information),
            "laplace_prior_precision": np.asarray(approximation.prior_precision),
            "electron_density_samples": density_samples,
            "electron_temperature_samples": temperature_samples,
        },
        scalars={"information_gain_bits": information_gain},
    )


def _mcmc_band(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    map_result: map_optimization_utils.MapResult,
    laplace: laplace_approximation_utils.LaplaceApproximation,
    rho: jt.Float[jax.Array, " n_rho"],
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None,
) -> BandOutcome:
    """NUTS started around the MAP's Laplace Gaussian, and the percentile band of its
    draws.

    The band comes off the same `band_from_unconstrained_draws` the two Gaussian
    recipes use, so the three differ in where their draws came from and nowhere else.
    """
    posterior = posterior_utils.build_posterior(
        inference_model_utils.bind_plasma_model(
            settings, priors, target, surrogate, fixed_coefficients
        ),
        map_result.point,
    )
    chain = mcmc_utils.run_mcmc(
        settings,
        priors,
        target,
        surrogate,
        map_result.point,
        posterior,
        # Both posteriors are built from the same model's sites, so they share
        # one coordinate order and the Laplace covariance needs no reordering.
        laplace.covariance,
        fixed_coefficients,
    )
    band, density_samples, temperature_samples = band_from_unconstrained_draws(
        settings, priors, posterior, chain.unconstrained_samples, rho
    )
    diagnostics = chain.diagnostics
    return BandOutcome(
        band=band,
        arrays={
            "unconstrained_samples": np.asarray(chain.unconstrained_samples),
            "mcmc_chain_samples": np.asarray(chain.chain_samples),
            "mcmc_warmup_chain_samples": np.asarray(chain.warmup_chain_samples),
            "mcmc_site_names": np.asarray(diagnostics.site_names),
            "mcmc_r_hat": np.asarray(diagnostics.r_hat),
            "mcmc_ess_bulk": np.asarray(diagnostics.ess_bulk),
            "mcmc_ess_tail": np.asarray(diagnostics.ess_tail),
            "mcmc_log_posterior_trace": np.asarray(chain.log_posterior_trace),
            "electron_density_samples": density_samples,
            "electron_temperature_samples": temperature_samples,
        },
        scalars={
            "mcmc_time_seconds": chain.seconds,
            "r_hat_max": diagnostics.r_hat_max,
            "ess_bulk_min": diagnostics.ess_bulk_min,
            "ess_tail_min": diagnostics.ess_tail_min,
            "n_divergences": diagnostics.n_divergences,
            "n_max_tree_depth": diagnostics.n_max_tree_depth,
            "max_tree_depth_fraction": diagnostics.max_tree_depth_fraction,
            "accept_prob_mean": diagnostics.accept_prob_mean,
            "step_size_mean": diagnostics.step_size_mean,
            "ebfmi_min": diagnostics.ebfmi_min,
            "n_leapfrog_steps": diagnostics.n_leapfrog_steps,
            "nuts_converged": diagnostics.converged,
        },
    )


def _constrained_samples(
    posterior: posterior_utils.Posterior,
    draws: jt.Float[jax.Array, "n_sample dim"],
    priors: priors_utils.ProfilePriors,
) -> tuple[np.ndarray, np.ndarray]:
    """Map unconstrained draws back to the two physical parameter matrices.

    One column at a time, `posterior.constrain` taking a single vector and not a batch.
    Which column holds which site comes from `posterior.site_names`, not index order.
    """
    columns = {
        name: jnp.asarray(posterior.transforms[name](draws[:, column]))
        for column, name in enumerate(posterior.site_names)
    }
    return (
        np.stack(
            [
                np.asarray(columns[name])
                for name in priors_utils.site_names(priors, "ne")
            ],
            axis=-1,
        ),
        np.stack(
            [
                np.asarray(columns[name])
                for name in priors_utils.site_names(priors, "te")
            ],
            axis=-1,
        ),
    )


def _store_profiles(
    arrays: dict[str, np.ndarray],
    prefix: str,
    profiles: profile_parametrization_types.PlasmaProfiles,
    rho: jt.Float[jax.Array, " n_rho"],
) -> None:
    """Add one profile set's `ne`, `Te` and pressure curves under a name prefix."""
    arrays[f"{prefix}_electron_density"] = np.asarray(
        profile_parametrization_types.electron_density(profiles, rho)
    )
    arrays[f"{prefix}_electron_temperature"] = np.asarray(
        profile_parametrization_types.electron_temperature(profiles, rho)
    )
    arrays[f"{prefix}_pressure"] = np.asarray(
        profile_parametrization_types.pressure_pascal(profiles, rho)
    )


def _posterior_from_samples(
    settings: inference_types.PlasmaEstimationSettings,
    density_samples: jt.Float[NpOrJaxArray, "n_sample n_ne"],
    temperature_samples: jt.Float[NpOrJaxArray, "n_sample n_te"],
    rho: jt.Float[jax.Array, " n_rho"],
) -> inference_types.PlasmaProfilePosterior:
    """Mean, median and percentile band of parameter draws, pushed through the profile
    map."""
    density = np.asarray(
        profile_parametrization_types.to_si(
            settings.profile_family,
            _evaluate_samples(settings, density_samples, rho),
            profile_parametrization_types.DENSITY_UNIT,
        )
    )
    temperature = np.asarray(
        profile_parametrization_types.to_si(
            settings.profile_family,
            _evaluate_samples(settings, temperature_samples, rho),
            profile_parametrization_types.TEMPERATURE_UNIT,
        )
    )
    pressure = 2.0 * thomson_scattering_utils.ELEMENTARY_CHARGE * density * temperature
    low, high = _BAND_PERCENTILES
    return inference_types.PlasmaProfilePosterior(
        rho=np.asarray(rho),
        electron_density_median=np.median(density, axis=0),
        electron_density_lower=np.percentile(density, low, axis=0),
        electron_density_upper=np.percentile(density, high, axis=0),
        electron_temperature_median=np.median(temperature, axis=0),
        electron_temperature_lower=np.percentile(temperature, low, axis=0),
        electron_temperature_upper=np.percentile(temperature, high, axis=0),
        pressure_median=np.median(pressure, axis=0),
        pressure_lower=np.percentile(pressure, low, axis=0),
        pressure_upper=np.percentile(pressure, high, axis=0),
        electron_density_mean=np.mean(density, axis=0),
        electron_temperature_mean=np.mean(temperature, axis=0),
        pressure_mean=np.mean(pressure, axis=0),
    )


def _evaluate_samples(
    settings: inference_types.PlasmaEstimationSettings,
    samples: jt.Float[NpOrJaxArray, "n_sample n_param"],
    rho: jt.Float[jax.Array, " n_rho"],
) -> jt.Float[jax.Array, "n_sample n_rho"]:
    """Evaluate one profile family on every draw, in the family's own units."""
    return jax.vmap(
        functools.partial(
            profile_parametrization_types.evaluate_profile,
            settings.profile_family,
            rho=rho,
        )
    )(jnp.asarray(samples))


def _likelihood_quantities(
    target: inference_target_types.InferenceTarget,
) -> list[str]:
    """Which observables this target compares against."""
    quantities = []
    if target.thomson_signals is not None:
        quantities.append("thomson_signals")
    if target.interferometer is not None:
        quantities.append("line_integrated_density")
    return quantities
