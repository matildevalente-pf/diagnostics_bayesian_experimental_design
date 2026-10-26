# One run's reported profiles and per-volume Thomson estimates as long tables, the rows
# `plots_plasma_state_estimation.plot_method_band_comparison` draws for every study.

import jax.numpy as jnp
import jax.scipy.stats
import numpy as np
import numpyro.distributions.transforms
import pandas as pd
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    inference_types,
    priors_utils,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E501
    inference_metrics_utils,
    inference_store_utils,
    synthetic_target_utils,
)

PROFILE_NAMES = {"ne": "electron_density", "Te": "electron_temperature"}
"""Each profile's table symbol, `ne` in m^-3 and `Te` in eV, and its record field."""

_MAP_EVALUATORS = {
    "ne": profile_parametrization_types.electron_density,
    "Te": profile_parametrization_types.electron_temperature,
}
"""How each profile is evaluated from the record's own MAP parameters."""


def profile_band_table(
    record: inference_types.PlasmaStateInferenceResult, arm: str
) -> pd.DataFrame:
    """One run's truth, initial guess, MAP and reported band, one row per `rho`.

    Args:
        record: The stored run.
        arm: The name its rows carry in the `arm` column.

    Returns:
        The rows, with the mean and the 5th, 50th and 95th percentiles of the band.
    """
    rho = np.asarray(record.inferred_profiles.rho)
    columns = {
        "profile_id": record.truth_name,
        "arm": arm,
        "method": record.settings.method,
        "rho": rho,
    }
    for symbol, name in PROFILE_NAMES.items():
        columns[f"{symbol}_truth"] = np.asarray(
            getattr(record.true_profiles, f"{name}_median")
        )
        columns[f"{symbol}_map"] = np.asarray(
            _MAP_EVALUATORS[symbol](record.map_profiles, rho)
        )
        columns[f"{symbol}_initial"] = np.asarray(
            getattr(record.initial_profiles, f"{name}_median")
        )
        columns[f"{symbol}_mean"] = inference_metrics_utils.central_curve(
            record.inferred_profiles, name
        )
        for quantile, field in (("q05", "lower"), ("q50", "median"), ("q95", "upper")):
            columns[f"{symbol}_{quantile}"] = np.asarray(
                getattr(record.inferred_profiles, f"{name}_{field}")
            )
    return pd.DataFrame(columns)


def thomson_points_table(
    record: inference_types.PlasmaStateInferenceResult,
    truth: desc_truth_types.DescTruth,
) -> pd.DataFrame:
    """Each scattering volume's own `ne` and `Te`, inverted from the signals `record`
    was fitted to, rebuilt from its seed and instrument; only the volumes it read.

    Args:
        record: The stored run whose signals are rebuilt.
        truth: The ground truth the run was generated from.

    Returns:
        One row per scattering volume, with its flux label and both widths.
    """
    target = synthetic_target_utils.build_synthetic_target(
        truth,
        inference_store_utils.load_placement(
            inference_store_utils.PLACEMENT_STORE, record.diagnostics_placement_id
        ),
        interferometer_n_sample=record.settings.interferometer_n_sample,
        trim_chords_to_plasma=record.settings.trim_chords_to_plasma,
        seed=record.settings.seed,
        add_noise=record.settings.add_measurement_noise,
        interferometer_sigma=record.settings.interferometer_sigma,
    )
    volume_rho, estimate = synthetic_target_utils.thomson_local_estimates(target, truth)
    volumes = (
        np.arange(np.asarray(volume_rho).shape[0])
        if record.thomson_volume_indices is None
        else np.asarray(record.thomson_volume_indices, dtype=int)
    )
    return pd.DataFrame(
        {
            "profile_id": record.truth_name,
            "volume_index": volumes,
            "rho": np.asarray(volume_rho)[volumes],
            "ne": np.asarray(estimate.electron_density)[volumes],
            "ne_sigma": np.asarray(estimate.electron_density_sigma)[volumes],
            "Te": np.asarray(estimate.electron_temperature_ev)[volumes],
            "Te_sigma": np.asarray(estimate.electron_temperature_sigma_ev)[volumes],
        }
    )


def calibration_table(
    record: inference_types.PlasmaStateInferenceResult,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Each inferred Thomson calibration correction `delta_j`: its MAP and Laplace band.

    Each `delta_j` is a monotonic map of one unconstrained coordinate, so its 5-95% band
    is exactly that map of the Gaussian marginal's quantiles.

    Args:
        record: The stored run.
        arrays: Its stored arrays, which carry the Laplace mean and covariance.

    Returns:
        One row per volume with a correction; empty when the run inferred none.
    """
    # The Laplace moments are stored in the posterior's site order, as the trajectories.
    names = [str(name) for name in arrays["trajectory_site_names"]]
    columns = [
        index
        for index, name in enumerate(names)
        if name.startswith(f"{priors_utils.CALIBRATION_PREFIX}_")
    ]
    bound = record.settings.thomson_calibration_bound
    prior = next(iter(priors_utils.calibration_priors([0], bound).values()))
    transform = numpyro.distributions.transforms.biject_to(prior.support)
    mean = np.asarray(arrays["laplace_mean"])[columns]
    sigma = np.sqrt(np.diag(np.asarray(arrays["laplace_covariance"])))[columns]
    z_95 = float(jax.scipy.stats.norm.ppf(0.95))
    return pd.DataFrame(
        {
            "profile_id": record.truth_name,
            "bound": bound,
            "volume_index": [int(names[index].split("_")[-1]) for index in columns],
            "map": np.asarray(transform(jnp.asarray(mean))),
            "q05": np.asarray(transform(jnp.asarray(mean - z_95 * sigma))),
            "q95": np.asarray(transform(jnp.asarray(mean + z_95 * sigma))),
        }
    ).sort_values("volume_index", ignore_index=True)
