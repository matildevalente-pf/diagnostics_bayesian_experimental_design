r"""What each Thomson uncertainty model costs the reconstruction, against the prior.

A comparison of the two noise models in the two-power family: every catalog truth is
measured with the complete diagnostic system under each model on `--n-seeds` noise
draws, one inference is run against each measurement set, and the draws of a truth and
model are pooled into one posterior. The prior is drawn from the same distributions the
model samples and pushed through the same profile evaluator, so all three appear on one
axis. The two inferred cases are posteriors, one per noise model, never likelihoods.

    python run_study.py --workers 7
    python run_study.py --truths wide_1_parametric hollow_gp --workers 2
    python run_study.py --skip-stored --workers 7        # resume an interrupted run
    python run_study.py --method mcmc_nuts --truths wide_1_parametric \
        wide_high_beta_gp wide_low_beta_gp --workers 6   # NUTS reference, from the MAP

The truths are independent and their cells run concurrently, one process each. A
cell differs from its siblings in its Thomson uncertainty model and in nothing else, the
truth, the seed, the layout and the surrogate all being shared. The model is a property
of the instrument, so it sets both the width the likelihood whitens by and the noise
draw itself, and the two arms of a truth therefore read different Thomson signals; the
chord readings are unaffected, the interferometer carrying its own uncertainty.

Writes `inference_ids.json` and the tables in `tables/` beside this script, and one
record per run into the shared `results_plots_tables/PlasmaStateInferenceResult/`.
`--method mcmc_nuts` writes `inference_ids_mcmc_nuts.json` and only the band table
`tables/prior_likelihood_posterior_profiles_mcmc_nuts.csv` instead.
"""

import argparse
import json
import pathlib
import re
import sys
import typing

_MODULE_DIRECTORY = pathlib.Path(__file__).resolve().parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parents[3]
sys.path.insert(0, str(_PACKAGE_DIRECTORY.parent))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxtyping as jt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E402,E501
    desc_truth_types,
    inference_run_utils,
    inference_types,
    priors_utils,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E402,E501
    diagnostic_design_point,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    inference_store_utils,
    sweep_utils,
    synthetic_target_utils,
)

PRIOR_CASE = "prior"
"""The case carrying no diagnostic likelihood at all, drawn rather than inferred."""

INFERENCE_CASES: tuple[thomson_scattering_utils.ThomsonNoiseModel, ...] = (
    "poisson",
    "electronic",
)
"""The Thomson uncertainty model each inferred case carries, which is also its name.

Every case reads the complete diagnostic system, so the arms differ in the uncertainty
`thomson_scattering_utils.signal_uncertainties` attaches to a reading and in nothing
else. A model is a property of the instrument, so it enters both the recorded signals
and the width the likelihood whitens by, and two arms are therefore different
instruments rather than two analyses of one measurement.
"""

WARM_START_CASE: thomson_scattering_utils.ThomsonNoiseModel = "electronic"
"""The case whose MAP a run of the other case starts from once its prior draws have
failed, since it reads the same plasma with uncertainties that L-BFGS can resolve."""

PROFILE_SYMBOLS = ("ne", "Te")
"""How the two profiles are named in the tables, `ne` in m^-3 and `Te` in eV."""

METHOD: inference_types.InferenceMethod = "map_laplace"
"""The recipe every inferred case runs: the MAP with the Laplace approximation at it."""

PROFILE_FAMILY: profile_parametrization_types.ProfileFamily = "two_power_zero_edge"
"""The family every case is expressed in, `center * (1 - rho^alpha)^beta` a profile."""

DEFAULT_N_PRIOR_SAMPLES = inference_run_utils.DEFAULT_N_POSTERIOR_SAMPLES
"""Prior draws, the same count the Laplace band is built from so the three cases'
quantiles carry the same Monte Carlo noise."""

PRIOR_SEED = 20260923
"""Seed of the prior draw, shared by every truth: the prior does not depend on one, so
one draw makes `U_ne_prior` and `U_Te_prior` a single number the ratios divide by."""

SEED_STRIDE = 1000
"""Seeds between consecutive truths, so no two truths ever share a noise draw."""

DEFAULT_N_SEEDS = 1
"""Noisy draws per truth and noise model; `--method mcmc_nuts` runs the first only."""

CREDIBLE_PERCENTILES = (5.0, 95.0)
"""The 5 to 95 percent credible interval every case reports, as `q05` and `q95`, the
same band every stored record and every other study of this area reports."""

STUDY_NAME = _MODULE_DIRECTORY.name
"""What this study's records carry in `study_name`, and what its manifest selects on."""

IDS_FILE = _MODULE_DIRECTORY / "inference_ids.json"
"""One row per stored run, written by `sweep_utils.write_manifest` off the records."""

TABLES_DIRECTORY = _MODULE_DIRECTORY / "tables"
"""Where the profile bands, the summary metrics and the configuration are written."""

PROFILES_FILE = TABLES_DIRECTORY / "prior_likelihood_posterior_profiles.csv"
"""Truth, MAP, initial guess, central estimate, spread and credible interval per truth,
case and `rho`."""

SUMMARY_FILE = TABLES_DIRECTORY / "prior_likelihood_posterior_summary.csv"
"""One row per truth and case: the RMSE, the integrated uncertainty and its ratio."""

CONFIGURATION_FILE = TABLES_DIRECTORY / "prior_likelihood_posterior_configuration.json"
"""What every run read, so the notebook can check the three cases share their data."""

THOMSON_POINTS_FILE = TABLES_DIRECTORY / "prior_likelihood_posterior_thomson_points.csv"
"""Each truth's per-volume Thomson `ne` and `Te`, the points the figures overlay."""

NUTS_CONFIGURATION = inference_types.McmcSettings(
    n_warmup=1000,
    n_samples=1000,
    n_chains=4,
    target_accept_prob=0.90,
    max_tree_depth=10,
    chain_init_width=3.0,
    init_strategy="laplace",
)
"""The sampler of `--method mcmc_nuts`, the `mcmc_benchmark` configuration with every
chain started at a draw of the MAP's Laplace Gaussian widened three times."""

NUTS_STUDY_NAME = f"{STUDY_NAME}_mcmc_nuts"
"""The `study_name` of a `--method mcmc_nuts` record, so the tables never load it."""

NUTS_IDS_FILE = _MODULE_DIRECTORY / "inference_ids_mcmc_nuts.json"
"""The manifest of the `--method mcmc_nuts` runs, labelled as their Laplace twins."""

NUTS_PROFILES_FILE = (
    TABLES_DIRECTORY / "prior_likelihood_posterior_profiles_mcmc_nuts.csv"
)
"""The columns of `PROFILES_FILE` for the NUTS runs, without the prior case."""


def main() -> None:
    arguments = _parse_arguments()
    sampled = arguments.method == "mcmc_nuts"
    study_name = NUTS_STUDY_NAME if sampled else STUDY_NAME
    ids_file = NUTS_IDS_FILE if sampled else IDS_FILE
    paths = sweep_utils.paths_from_arguments(arguments, study_name)
    if not arguments.tables_only:
        # The warm-start case runs first, so a run of the other case whose prior draws
        # all fail can start once more from its MAP, same truth and same seed.
        for case in dict.fromkeys((WARM_START_CASE, *INFERENCE_CASES)):
            needs_stored = arguments.skip_stored or case != WARM_START_CASE
            stored = stored_runs(ids_file) if needs_stored else {}
            cells = [
                sweep_utils.SweepCell(
                    name=f"{truth_name}_{case}_seed{seed}",
                    label=f"{truth_name:<16s} {case} seed {seed}",
                    truth_name=truth_name,
                    settings=_settings(seed, arguments.method),
                    design_point=diagnostic_design_point.DiagnosticDesignPoint(
                        thomson_noise_model=case
                    ),
                    warm_start=(
                        _map_point(stored[(truth_name, WARM_START_CASE, seed)])
                        if case != WARM_START_CASE
                        and (truth_name, WARM_START_CASE, seed) in stored
                        else None
                    ),
                )
                for truth_name in arguments.truths
                for seed in truth_seeds(arguments.seed, truth_name, arguments.n_seeds)[
                    : 1 if sampled else None
                ]
                if not (arguments.skip_stored and (truth_name, case, seed) in stored)
            ]
            print(
                f"[sweep] {case}: {len(cells)} cells, {len(arguments.truths)} truths x "
                f"{1 if sampled else arguments.n_seeds} draws in the {PROFILE_FAMILY} "
                f"family, {arguments.method}",
                flush=True,
            )
            sweep_utils.run_sweep(cells, paths, workers=arguments.workers)
            sweep_utils.write_manifest(ids_file, study_name)
    if sampled:
        write_sampled_profiles(arguments.truths)
    else:
        write_tables(arguments.truths, arguments.n_prior_samples, paths)


def write_tables(
    truth_names: list[str], n_prior_samples: int, paths: sweep_utils.SweepPaths
) -> None:
    """Write the profile bands, the metrics, the Thomson points and the configuration.

    Args:
        truth_names: The truths to report, each needing both inferred cases stored.
        n_prior_samples: Draws taken from the prior and propagated to profiles.
        paths: Where the dataset the truths are read back from lives.
    """
    # Every draw of a truth and case, in seed order, pooled into one posterior below.
    records: dict[tuple[str, str], list[tuple]] = {}
    for (truth_name, case, _), identifier in sorted(stored_runs(IDS_FILE).items()):
        records.setdefault((truth_name, case), []).append(
            inference_store_utils.load_inference(
                inference_store_utils.INFERENCE_STORE, identifier
            )
        )
    complete = [
        name
        for name in truth_names
        if all((name, case) in records for case in INFERENCE_CASES)
    ]
    missing = sorted(set(truth_names) - set(complete))
    if missing:
        print(f"[tables] incomplete and left out: {', '.join(missing)}")
    if not complete:
        print("[tables] nothing stored yet; run the sweep first")
        return

    reference, reference_arrays = records[(complete[0], INFERENCE_CASES[-1])][0]
    rho = np.asarray(reference_arrays["rho"])
    rho_grid = jnp.asarray(rho)
    prior_density, prior_temperature = prior_profile_curves(
        reference.settings, rho_grid, n_prior_samples, PRIOR_SEED
    )
    blocks = []
    for truth_name in complete:
        truth = records[(truth_name, INFERENCE_CASES[-1])][0][0].true_profiles
        for case in (PRIOR_CASE, *INFERENCE_CASES):
            point_columns = {}
            if case == PRIOR_CASE:
                density, temperature = prior_density, prior_temperature
            else:
                # The first draw's MAP and the prior draw it started from, the point
                # estimates a single-draw run has.
                first_record = records[(truth_name, case)][0][0]
                point_columns = {
                    "ne_map": np.asarray(
                        profile_parametrization_types.electron_density(
                            first_record.map_profiles, rho_grid
                        )
                    ),
                    "Te_map": np.asarray(
                        profile_parametrization_types.electron_temperature(
                            first_record.map_profiles, rho_grid
                        )
                    ),
                    "ne_initial": np.asarray(
                        first_record.initial_profiles.electron_density_median
                    ),
                    "Te_initial": np.asarray(
                        first_record.initial_profiles.electron_temperature_median
                    ),
                }
                density, temperature = (
                    np.concatenate(
                        [
                            inference_run_utils.profile_samples_in_si(
                                record.settings,
                                name,
                                arrays[f"{name}_samples"],
                                rho_grid,
                            )
                            for record, arrays in records[(truth_name, case)]
                        ]
                    )
                    for name in ("electron_density", "electron_temperature")
                )
            blocks.append(
                pd.DataFrame(
                    {
                        "profile_id": truth_name,
                        "inference_case": case,
                        "rho": rho,
                        "ne_truth": np.asarray(truth.electron_density_median),
                        "Te_truth": np.asarray(truth.electron_temperature_median),
                        **point_columns,
                        **profile_statistics("ne", density),
                        **profile_statistics("Te", temperature),
                    }
                )
            )
    profiles = pd.concat(blocks, ignore_index=True)
    TABLES_DIRECTORY.mkdir(exist_ok=True)
    profiles.to_csv(PROFILES_FILE, index=False)
    summary_table(profiles).to_csv(SUMMARY_FILE, index=False)
    thomson_points_table(complete, records, paths).to_csv(
        THOMSON_POINTS_FILE, index=False
    )
    CONFIGURATION_FILE.write_text(
        json.dumps(configuration(complete, records, rho, n_prior_samples), indent=4)
        + "\n"
    )
    for path in (PROFILES_FILE, SUMMARY_FILE, THOMSON_POINTS_FILE, CONFIGURATION_FILE):
        print(f"[tables] {path}")


def write_sampled_profiles(truth_names: list[str]) -> None:
    """Write the band table of the `--method mcmc_nuts` runs, as `write_tables` does.

    Args:
        truth_names: The truths to report; a truth and case with no NUTS run is skipped.
    """
    # NUTS runs each truth's first draw only.
    records = {}
    for (truth_name, case, _), identifier in sorted(stored_runs(NUTS_IDS_FILE).items()):
        records.setdefault(
            (truth_name, case),
            inference_store_utils.load_inference(
                inference_store_utils.INFERENCE_STORE, identifier
            ),
        )
    blocks = []
    for truth_name in truth_names:
        for case in INFERENCE_CASES:
            if (truth_name, case) not in records:
                continue
            record, arrays = records[(truth_name, case)]
            rho = np.asarray(arrays["rho"])
            density, temperature = (
                inference_run_utils.profile_samples_in_si(
                    record.settings, name, arrays[f"{name}_samples"], jnp.asarray(rho)
                )
                for name in ("electron_density", "electron_temperature")
            )
            blocks.append(
                pd.DataFrame(
                    {
                        "profile_id": truth_name,
                        "inference_case": case,
                        "rho": rho,
                        "ne_truth": np.asarray(
                            record.true_profiles.electron_density_median
                        ),
                        "Te_truth": np.asarray(
                            record.true_profiles.electron_temperature_median
                        ),
                        **profile_statistics("ne", density),
                        **profile_statistics("Te", temperature),
                    }
                )
            )
    if not blocks:
        print("[tables] no NUTS run stored yet; run with --method mcmc_nuts first")
        return
    TABLES_DIRECTORY.mkdir(exist_ok=True)
    pd.concat(blocks, ignore_index=True).to_csv(NUTS_PROFILES_FILE, index=False)
    print(f"[tables] {NUTS_PROFILES_FILE}")


def prior_profile_curves(
    settings: inference_types.PlasmaEstimationSettings,
    rho: jt.Float[jax.Array, " n_rho"],
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the profile priors and push the draws through the family's own evaluator.

    The same distributions `inference_model_utils.plasma_model` samples and the same
    `profile_samples_in_si` the posterior bands come off, so a prior band differs from a
    posterior band in where its draws came from and nowhere else.

    Args:
        settings: The run configuration, supplying the family the priors belong to.
        rho: Radial grid to evaluate the drawn profiles on.
        n_samples: Draws to take.
        seed: Seed of the draw.

    Returns:
        The density and the temperature sample curves, in m^-3 and eV.
    """
    priors = priors_utils.default_priors(
        settings.profile_family, calibration_bound=settings.thomson_calibration_bound
    )
    names = sorted(priors)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(names))
    columns = {
        name: jnp.asarray(priors[name].sample(key, (n_samples,)))
        for name, key in zip(names, keys)
    }
    density, temperature = (
        inference_run_utils.profile_samples_in_si(
            settings,
            name,
            priors_utils.stack_profile_columns(columns, priors, prefix),
            rho,
        )
        for name, prefix in (
            ("electron_density", "ne"),
            ("electron_temperature", "te"),
        )
    )
    return density, temperature


def thomson_points_table(
    truth_names: list[str],
    records: dict[tuple[str, str], list[tuple]],
    paths: sweep_utils.SweepPaths,
) -> pd.DataFrame:
    """Each truth's per-volume Thomson `ne` and `Te`, inverted from its own signals, for
    every noise draw.

    One block per arm, since a noise model changes both the recorded signals and the
    width attached to them, so the points and their bars belong to the arm they were
    inverted from.

    Args:
        truth_names: The truths to invert.
        records: Every draw's `(record, arrays)`, keyed by truth and case.
        paths: Where the dataset the truths are read back from lives.

    Returns:
        One row per truth, arm, draw and scattering volume, with the flux label and
        both widths.
    """
    parquet_path = paths.data_directory / f"{paths.dataset}.parquet"
    settings_path = paths.data_directory / f"{paths.dataset}_settings.json"
    blocks = []
    for truth_name in truth_names:
        truth = desc_truth_types.read_desc_truth(
            truth_name, parquet_path, settings_path
        )
        for case in INFERENCE_CASES:
            for record, _ in records[(truth_name, case)]:
                blocks.append(_thomson_points(truth, case, record))
    return pd.concat(blocks, ignore_index=True)


def _thomson_points(
    truth: desc_truth_types.DescTruth,
    case: str,
    record: inference_types.PlasmaStateInferenceResult,
) -> pd.DataFrame:
    """One draw's per-volume Thomson `ne` and `Te`, rebuilt from its own provenance, its
    seed and the stored instrument, so these are the signals it was fitted to."""
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
    return pd.DataFrame(
        {
            "profile_id": truth.name,
            "inference_case": case,
            "seed": record.settings.seed,
            "volume_index": np.arange(np.asarray(volume_rho).shape[0]),
            "rho": np.asarray(volume_rho),
            "ne": np.asarray(estimate.electron_density),
            "ne_sigma": np.asarray(estimate.electron_density_sigma),
            "Te": np.asarray(estimate.electron_temperature_ev),
            "Te_sigma": np.asarray(estimate.electron_temperature_sigma_ev),
        }
    )


def profile_statistics(symbol: str, curves: np.ndarray) -> dict[str, np.ndarray]:
    """Central estimate, spread and 5-95% interval of one set of sample curves.

    Args:
        symbol: How the profile is named in the table, an entry of `PROFILE_SYMBOLS`.
        curves: One profile curve per draw, in SI.

    Returns:
        The five per-`rho` columns that profile contributes to the table.
    """
    low, high = CREDIBLE_PERCENTILES
    return {
        f"{symbol}_mean": np.mean(curves, axis=0),
        f"{symbol}_median": np.median(curves, axis=0),
        f"{symbol}_std": np.std(curves, axis=0, ddof=1),
        f"{symbol}_q05": np.percentile(curves, low, axis=0),
        f"{symbol}_q95": np.percentile(curves, high, axis=0),
    }


def summary_table(profiles: pd.DataFrame) -> pd.DataFrame:
    """Per truth and case: the RMSE of the mean, the reported profile, the integrated
    uncertainty `U`, its ratio `R = U / U_prior`, and the fraction of the `rho` grid on
    which the 5-95% band covers the truth.

    Args:
        profiles: The long band table, as `write_tables` assembled it.

    Returns:
        One row per truth and case, the prior's row carrying `R = 1` by construction.
    """
    rows = []
    for profile_id in profiles["profile_id"].unique():
        for case in (PRIOR_CASE, *INFERENCE_CASES):
            block = profiles.loc[
                (profiles["profile_id"] == profile_id)
                & (profiles["inference_case"] == case)
            ]
            rho = block["rho"].to_numpy()
            entry: dict[str, object] = {
                "profile_id": profile_id,
                "inference_case": case,
            }
            for symbol in PROFILE_SYMBOLS:
                truth = block[f"{symbol}_truth"].to_numpy()
                deviation = block[f"{symbol}_mean"].to_numpy() - truth
                entry[f"{symbol}_RMSE"] = float(np.sqrt(np.mean(np.square(deviation))))
                entry[f"U_{symbol}"] = float(
                    np.trapezoid(block[f"{symbol}_std"].to_numpy(), rho)
                )
                entry[f"coverage_{symbol}"] = float(
                    np.mean(
                        (block[f"{symbol}_q05"].to_numpy() <= truth)
                        & (truth <= block[f"{symbol}_q95"].to_numpy())
                    )
                )
            rows.append(entry)
    table = pd.DataFrame(rows)
    for symbol in PROFILE_SYMBOLS:
        prior = table.loc[table["inference_case"] == PRIOR_CASE].set_index("profile_id")
        table[f"R_{symbol}"] = (
            table[f"U_{symbol}"].to_numpy()
            / prior[f"U_{symbol}"].reindex(table["profile_id"]).to_numpy()
        )
    return table.reindex(
        columns=["profile_id", "inference_case"]
        + [f"{symbol}_RMSE" for symbol in PROFILE_SYMBOLS]
        + [f"U_{symbol}" for symbol in PROFILE_SYMBOLS]
        + [f"R_{symbol}" for symbol in PROFILE_SYMBOLS]
        + [f"coverage_{symbol}" for symbol in PROFILE_SYMBOLS]
    )


def configuration(
    truth_names: list[str],
    records: dict[tuple[str, str], list[tuple]],
    rho: np.ndarray,
    n_prior_samples: int,
) -> dict:
    """What every run read, so the notebook can check the cases share their data.

    The noise draw is a deterministic function of the truth, the instrument and the
    seed, so two cells agreeing on those three read identical chord readings; the index
    lists say which channels the likelihood could see.

    Args:
        truth_names: The truths reported.
        records: Every draw's `(record, arrays)`, keyed by truth and case.
        rho: The reporting grid every case was evaluated on.
        n_prior_samples: Draws behind the prior band.

    Returns:
        The configuration, written beside the tables as JSON.
    """
    runs = []
    for truth_name in truth_names:
        for case in INFERENCE_CASES:
            for record, _ in records[(truth_name, case)]:
                runs.append(_run_summary(truth_name, case, record))
    return {
        "study_name": STUDY_NAME,
        "profile_family": PROFILE_FAMILY,
        "method": METHOD,
        "prior_case": PRIOR_CASE,
        "inference_cases": list(INFERENCE_CASES),
        "n_prior_samples": n_prior_samples,
        "n_posterior_samples": inference_run_utils.DEFAULT_N_POSTERIOR_SAMPLES,
        "prior_seed": PRIOR_SEED,
        "n_rho": int(rho.shape[0]),
        "runs": runs,
    }


def _run_summary(
    truth_name: str, case: str, record: inference_types.PlasmaStateInferenceResult
) -> dict:
    """What one run read and how its MAP ended, for `configuration`."""
    return {
        "profile_id": truth_name,
        "inference_case": case,
        "inference_id": record.inference_id,
        "noise_seed": record.settings.seed,
        "n_map_starts": record.settings.map.n_starts,
        "n_map_starts_converged": record.n_map_starts_converged,
        "map_converged": record.map_converged,
        "n_map_modes": record.n_map_modes,
        "diagnostics_placement_id": record.diagnostics_placement_id,
        "surrogate_name": record.surrogate_name,
        "thomson_volume_indices": record.thomson_volume_indices,
        "interferometer_chord_indices": record.interferometer_chord_indices,
        "log_posterior_map": record.log_posterior_map,
        "map_log_posterior_spread": record.map_log_posterior_spread,
        "map_gradient_inf_norm": record.map_gradient_inf_norm,
        "map_newton_decrement": record.map_newton_decrement,
        "chi_squared": record.chi_squared,
        "n_degrees_of_freedom": record.n_degrees_of_freedom,
    }


def stored_runs(ids_file: pathlib.Path) -> dict[tuple[str, str, int], str]:
    """Every run of a manifest, keyed by truth, noise model and noise seed.

    Args:
        ids_file: The manifest to read.

    Returns:
        The inference ids; the seed is read off each record, so a run labelled before
        the draws were counted is found at its own seed like any other.
    """
    runs = {}
    for label, identifier in sweep_utils.current_ids(ids_file).items():
        record = inference_store_utils.load_record(
            inference_store_utils.INFERENCE_STORE, identifier
        )
        case = re.sub(r"_seed\d+$", "", label.removeprefix(f"{record.truth_name}_"))
        runs[(record.truth_name, case, record.settings.seed)] = identifier
    return runs


def truth_seeds(base_seed: int, truth_name: str, n_seeds: int) -> list[int]:
    """The noise seeds of one truth, which both of its inference cases share.

    Args:
        base_seed: The study's `--seed`.
        truth_name: Which truth the draws are for.
        n_seeds: How many draws that truth gets.

    Returns:
        `n_seeds` consecutive seeds, offset by the truth's catalog position so no two
        truths share a draw.
    """
    offset = base_seed + SEED_STRIDE * sorted(desc_truth_types.DESC_TRUTH_IDS).index(
        truth_name
    )
    return [offset + replicate for replicate in range(n_seeds)]


def _map_point(inference_id: str) -> dict[str, float]:
    """The reported MAP of a stored run, one value per sample site: the end point of its
    best converged attempt, or of its best attempt when none converged."""
    _, arrays = inference_store_utils.load_inference(
        inference_store_utils.INFERENCE_STORE, inference_id
    )
    converged = arrays["start_converged"]
    candidates = np.flatnonzero(
        converged if converged.any() else np.ones_like(converged)
    )
    best = int(candidates[np.argmax(arrays["start_log_posterior"][candidates])])
    return {
        str(name): float(value)
        for name, value in zip(
            arrays["trajectory_site_names"],
            arrays["start_trajectories"][best, -1],
            strict=True,
        )
    }


def _settings(
    seed: int, method: inference_types.InferenceMethod
) -> inference_types.PlasmaEstimationSettings:
    """One cell's configuration; only the seed varies, between truths and draws."""
    return inference_types.PlasmaEstimationSettings(
        method=method,
        mcmc=(
            NUTS_CONFIGURATION
            if method == "mcmc_nuts"
            else inference_types.McmcSettings()
        ),
        profile_family=PROFILE_FAMILY,
        seed=seed,
        add_measurement_noise=True,
        verbose=False,
    )


def _parse_arguments() -> argparse.Namespace:
    """The study's command line; the ablated quantity is which diagnostics are read."""
    defaults = inference_types.PlasmaEstimationSettings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--truths",
        nargs="+",
        default=sorted(desc_truth_types.DESC_TRUTH_IDS),
        choices=sorted(desc_truth_types.DESC_TRUTH_IDS),
        help="Ground truths every case is run on.",
    )
    parser.add_argument(
        "--method",
        default=METHOD,
        choices=list(typing.get_args(inference_types.InferenceMethod)),
        help=(
            "Recipe every cell runs; `mcmc_nuts` stores under its own study name and "
            "manifest and writes only its band table."
        ),
    )
    parser.add_argument(
        "--n-prior-samples",
        type=int,
        default=DEFAULT_N_PRIOR_SAMPLES,
        help="Draws taken from the prior and propagated to profiles.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Base seed; each truth's noise draws are `truth_seeds` of it, and both "
        "cases share them, which is what makes them comparable.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Noisy draws per truth and noise model; mcmc_nuts runs the first only.",
    )
    parser.add_argument(
        "--skip-stored",
        action="store_true",
        help="Skip cells the manifest already has a current run for.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Skip the sweep and rewrite the tables off the stored records.",
    )
    sweep_utils.add_sweep_arguments(parser)
    parser.set_defaults(workers=len(desc_truth_types.DESC_TRUTH_IDS))
    return parser.parse_args()


if __name__ == "__main__":
    main()
