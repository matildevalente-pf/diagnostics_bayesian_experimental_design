r"""Benchmark MAP with the Laplace approximation against NUTS, truth by truth.

Two recipes, `map_laplace` and `mcmc_nuts`, in the two-power family. `map_laplace` runs
on `--n-seeds` noisy draws of every catalog truth and the sampler on the first of them,
so the recipes are compared on that shared draw and a difference is the recipe alone.
The sampler is only a reference where its chains converged, which is why every run
reports the diagnostics of `mcmc_convergence_utils` and a PASS or FAIL.

    python run_study.py --arms map_laplace --workers 7      # the cheap arm, minutes
    python run_study.py --arms mcmc_nuts --workers 7        # the reference, hours
    python run_study.py --skip-stored --workers 7           # resume an interrupted run
    python run_study.py --tables-only                       # rewrite the tables

Writes `inference_ids.json` and the tables in `tables/` beside this script, and one
record per run into the shared `results_plots_tables/PlasmaStateInferenceResult/`.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E402,E501
    desc_truth_types,
    inference_types,
    mcmc_convergence_utils,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E402,E501
    diagnostic_design_point,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    inference_metrics_utils,
    inference_store_utils,
    profile_table_utils,
    sweep_utils,
)

NUTS_CONFIGURATION = inference_types.McmcSettings(
    n_warmup=1000,
    n_samples=1000,
    n_chains=4,
    target_accept_prob=0.90,
    max_tree_depth=10,
    chain_init_width=3.0,
    dense_mass=True,
)
"""The sampler of this benchmark: NumPyro's step-size and dense mass-matrix adaptation
over the warmup, kept for the trace figure alone, no thinning, and each chain started at
its own draw of the MAP's Laplace Gaussian widened three times."""


class Arm(typing.NamedTuple):
    """One column of the benchmark: a recipe, and where its chains start."""

    method: inference_types.InferenceMethod
    """The recipe the cell runs."""

    init_strategy: inference_types.ChainInitStrategy
    """Where a sampled arm starts its chains; ignored by `map_laplace`."""


ARMS: dict[str, Arm] = {
    "map_laplace": Arm("map_laplace", "map"),
    "mcmc_nuts": Arm("mcmc_nuts", "laplace"),
}
"""The columns compared, keyed by the name their cells and table columns carry."""

PROFILE_FAMILY: profile_parametrization_types.ProfileFamily = "two_power_zero_edge"
"""The family both recipes infer in, `center * (1 - rho^alpha)^beta` per profile."""

NOISE_MODEL: thomson_scattering_utils.ThomsonNoiseModel = "electronic"
"""The Thomson uncertainty both recipes carry, the W7-X detector noise curve
(`thomson_scattering_utils.signal_uncertainties`)."""

SAMPLER_COLUMNS = (
    "r_hat_max",
    "ess_bulk_min",
    "ess_tail_min",
    "n_divergences",
    "n_max_tree_depth",
    "max_tree_depth_fraction",
    "accept_prob_mean",
    "ebfmi_min",
    "n_leapfrog_steps",
    "nuts_converged",
)
"""The summary columns only a sampled run fills, which are the record fields of the
same name; nan in a table written before the sampler has run."""

SEED_STRIDE = 1000
"""Seeds between consecutive truths, so no two truths share a noise draw."""

DEFAULT_N_SEEDS = 1
"""Noisy draws per truth for `map_laplace`; a sampled arm runs the first draw only,
at 2 to 3 h per run."""

STUDY_NAME = _MODULE_DIRECTORY.name
"""What this study's records carry in `study_name`, and what its manifest selects on."""

IDS_FILE = _MODULE_DIRECTORY / "inference_ids.json"
"""One row per stored run, written by `sweep_utils.write_manifest` off the records."""

TABLES_DIRECTORY = _MODULE_DIRECTORY / "tables"
"""Where the four tables and the configuration JSON are written."""

PROFILES_FILE = TABLES_DIRECTORY / "mcmc_benchmark_profiles.csv"
"""Truth and both recipes' `mean`, `q05`, `q50` and `q95` per truth, method and
`rho`."""

SUMMARY_FILE = TABLES_DIRECTORY / "mcmc_benchmark_summary.csv"
"""One row per truth: both recipes' error, width and cost, and the sampler's
convergence diagnostics with its PASS or FAIL."""

SITE_DIAGNOSTICS_FILE = TABLES_DIRECTORY / "mcmc_benchmark_site_diagnostics.csv"
"""`r_hat` and both effective sample sizes of every inferred parameter of every
sampled run."""

THOMSON_POINTS_FILE = TABLES_DIRECTORY / "mcmc_benchmark_thomson_points.csv"
"""Each truth's per-volume Thomson `ne` and `Te`, the points the figures overlay."""

CONFIGURATION_FILE = TABLES_DIRECTORY / "mcmc_benchmark_configuration.json"
"""The sampler configuration, the convergence criteria, and what every run read."""


def main() -> None:
    arguments = _parse_arguments()
    paths = sweep_utils.paths_from_arguments(arguments, STUDY_NAME)
    if not arguments.tables_only:
        stored = set(stored_runs()) if arguments.skip_stored else set()
        cells = [
            sweep_utils.SweepCell(
                name=f"{truth_name}_{arm_name}_seed{seed}",
                label=f"{truth_name:<16s} {arm_name} seed {seed}",
                truth_name=truth_name,
                settings=_settings(arguments, ARMS[arm_name], seed),
                design_point=diagnostic_design_point.DiagnosticDesignPoint(
                    thomson_noise_model=NOISE_MODEL
                ),
            )
            for truth_name in arguments.truths
            for arm_name in arguments.arms
            for seed in cell_seeds(arguments.seed, truth_name, arguments.n_seeds)[
                : 1 if ARMS[arm_name].method == "mcmc_nuts" else None
            ]
            if (truth_name, arm_name, seed) not in stored
        ]
        print(
            f"[sweep] {len(cells)} cells: {len(arguments.arms)} arms x "
            f"{len(arguments.truths)} truths, {arguments.n_seeds} draws for "
            f"map_laplace and the first for NUTS, in the {PROFILE_FAMILY} family, "
            f"Thomson noise {NOISE_MODEL}"
            + (f", {len(stored)} already stored and skipped" if stored else ""),
            flush=True,
        )
        sweep_utils.run_sweep(cells, paths, workers=arguments.workers)
        sweep_utils.write_manifest(IDS_FILE, STUDY_NAME)
    write_tables(paths)


def write_tables(paths: sweep_utils.SweepPaths) -> None:
    """Write the bands, the summary, the per-parameter diagnostics and the points.

    Every number comes off the stored records, so a table cannot disagree with the run
    it reports; only the Thomson points are rebuilt, from each run's own seed.

    Args:
        paths: Where the dataset the truths are read back from lives.
    """
    runs = stored_runs()
    # The recipes are compared on each truth's first draw, the only one NUTS ran.
    first = {
        truth: min(seed for (name, _, seed) in runs if name == truth)
        for truth, _, _ in runs
    }
    records = {
        (truth, arm): record
        for (truth, arm, seed), record in runs.items()
        if seed == first[truth]
    }
    truths = [
        name
        for name in sorted(desc_truth_types.DESC_TRUTH_IDS)
        if any((name, arm_name) in records for arm_name in ARMS)
    ]
    if not truths:
        print("[tables] nothing stored yet; run the sweep first")
        return
    for truth_name in truths:
        absent = [name for name in ARMS if (truth_name, name) not in records]
        if absent:
            print(f"[tables] {truth_name} has no run of: {', '.join(absent)}")

    profiles = profile_table(truths, records)
    TABLES_DIRECTORY.mkdir(exist_ok=True)
    profiles.to_csv(PROFILES_FILE, index=False)
    summary_table(truths, records, profiles).to_csv(SUMMARY_FILE, index=False)
    site_diagnostics_table(truths).to_csv(SITE_DIAGNOSTICS_FILE, index=False)
    thomson_points_table(truths, records, paths).to_csv(
        THOMSON_POINTS_FILE, index=False
    )
    CONFIGURATION_FILE.write_text(
        json.dumps(configuration(truths, records), indent=4) + "\n"
    )
    for path in (
        PROFILES_FILE,
        SUMMARY_FILE,
        SITE_DIAGNOSTICS_FILE,
        THOMSON_POINTS_FILE,
        CONFIGURATION_FILE,
    ):
        print(f"[tables] {path}")


def profile_table(
    truths: list[str],
    records: dict[tuple[str, str], inference_types.PlasmaStateInferenceResult],
) -> pd.DataFrame:
    """Both recipes' reported band and the truth, one row per truth, method and `rho`.

    The curves are the record's own `inferred_profiles`, which every recipe fills with
    the mean and the 5th, 50th and 95th percentiles of its draws in profile space.

    Args:
        truths: The truths to report.
        records: The stored records, keyed by truth and method.

    Returns:
        The long band table.
    """
    blocks = [
        profile_table_utils.profile_band_table(
            records[(truth_name, arm_name)], arm_name
        )
        for truth_name in truths
        for arm_name in ARMS
        if (truth_name, arm_name) in records
    ]
    return pd.concat(blocks, ignore_index=True)


def summary_table(
    truths: list[str],
    records: dict[tuple[str, str], inference_types.PlasmaStateInferenceResult],
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """One row per truth, holding both recipes and the sampler's convergence verdict.

    The error is `inference_metrics_utils.profile_accuracy`, the RMSE of the reported
    mean against the truth as a percentage of the truth's peak, which is what every
    other study of this area reports.

    Args:
        truths: The truths to report.
        records: The stored records, keyed by truth and method.
        profiles: The long band table, which the widths are read from.

    Returns:
        The summary, one row per truth.
    """
    rows = []
    for truth_name in truths:
        entry: dict[str, object] = {"profile_id": truth_name}
        # A sampled arm's columns exist whether or not it has run, so a table written
        # before it does still has the shape the notebook reads.
        for arm_name, arm in ARMS.items():
            if arm.method == "mcmc_nuts":
                entry.update(
                    {f"{name}_{arm_name}": float("nan") for name in SAMPLER_COLUMNS}
                )
        for arm_name, arm in ARMS.items():
            record = records.get((truth_name, arm_name))
            if record is None:
                continue
            accuracy = {
                item.name: item
                for item in inference_metrics_utils.profile_accuracy(record)
            }
            block = profiles.loc[
                (profiles["profile_id"] == truth_name) & (profiles["arm"] == arm_name)
            ]
            for symbol, name in profile_table_utils.PROFILE_NAMES.items():
                entry[f"{symbol}_rmse_percent_{arm_name}"] = accuracy[
                    name
                ].error_percent
                entry[f"{symbol}_width_{arm_name}"] = float(
                    np.trapezoid(
                        block[f"{symbol}_q95"].to_numpy()
                        - block[f"{symbol}_q05"].to_numpy(),
                        block["rho"].to_numpy(),
                    )
                )
            entry[f"seconds_{arm_name}"] = record.total_time_seconds
            if arm.method == "mcmc_nuts":
                entry.update(
                    {
                        f"{name}_{arm_name}": getattr(record, name)
                        for name in SAMPLER_COLUMNS
                    }
                )
        rows.append(entry)
    return pd.DataFrame(rows)


def site_diagnostics_table(truths: list[str]) -> pd.DataFrame:
    """Every sampled parameter's `r_hat` and effective sample sizes, truth by truth.

    Args:
        truths: The truths to report, those without a sampled run left out.

    Returns:
        One row per truth and inferred parameter; empty when nothing was sampled.
    """
    sampled = {name for name, arm in ARMS.items() if arm.method == "mcmc_nuts"}
    blocks = []
    for label, identifier in sweep_utils.current_ids(IDS_FILE).items():
        record, arrays = inference_store_utils.load_inference(
            inference_store_utils.INFERENCE_STORE, identifier
        )
        arm_name = arm_of(label, record.truth_name)
        if arm_name not in sampled or record.truth_name not in truths:
            continue
        blocks.append(
            pd.DataFrame(
                {
                    "profile_id": record.truth_name,
                    "arm": arm_name,
                    "site": [str(name) for name in arrays["mcmc_site_names"]],
                    "r_hat": arrays["mcmc_r_hat"],
                    "ess_bulk": arrays["mcmc_ess_bulk"],
                    "ess_tail": arrays["mcmc_ess_tail"],
                }
            )
        )
    if not blocks:
        return pd.DataFrame(
            columns=["profile_id", "arm", "site", "r_hat", "ess_bulk", "ess_tail"]
        )
    return pd.concat(blocks, ignore_index=True)


def thomson_points_table(
    truths: list[str],
    records: dict[tuple[str, str], inference_types.PlasmaStateInferenceResult],
    paths: sweep_utils.SweepPaths,
) -> pd.DataFrame:
    """Each truth's per-volume Thomson `ne` and `Te`, inverted from its own signals.

    Rebuilt from a run's own seed and stored instrument, so these are the signals both
    recipes of that truth were fitted to, each volume inverted on its own.

    Args:
        truths: The truths to invert.
        records: The stored records, keyed by truth and method.
        paths: Where the dataset the truths are read back from lives.

    Returns:
        One row per truth and scattering volume, with the flux label and both widths.
    """
    parquet_path = paths.data_directory / f"{paths.dataset}.parquet"
    settings_path = paths.data_directory / f"{paths.dataset}_settings.json"
    blocks = []
    for truth_name in truths:
        record = next(
            records[(truth_name, arm_name)]
            for arm_name in ARMS
            if (truth_name, arm_name) in records
        )
        truth = desc_truth_types.read_desc_truth(
            truth_name, parquet_path, settings_path
        )
        blocks.append(profile_table_utils.thomson_points_table(record, truth))
    return pd.concat(blocks, ignore_index=True)


def configuration(
    truths: list[str],
    records: dict[tuple[str, str], inference_types.PlasmaStateInferenceResult],
) -> dict:
    """What the sampler was configured with, what it is judged by, and what ran.

    Args:
        truths: The truths reported.
        records: The stored records, keyed by truth and method.

    Returns:
        The configuration, written beside the tables as JSON.
    """
    runs = []
    for truth_name in truths:
        for arm_name in ARMS:
            record = records.get((truth_name, arm_name))
            if record is None:
                continue
            runs.append(
                {
                    "profile_id": truth_name,
                    "arm": arm_name,
                    "method": record.settings.method,
                    "chain_init_strategy": record.settings.mcmc.init_strategy,
                    "inference_id": record.inference_id,
                    "noise_seed": record.settings.seed,
                    "n_posterior_samples": record.settings.n_posterior_samples,
                    "diagnostics_placement_id": record.diagnostics_placement_id,
                    "surrogate_name": record.surrogate_name,
                    "n_map_starts_converged": record.n_map_starts_converged,
                    "map_converged": record.map_converged,
                    "n_map_modes": record.n_map_modes,
                    "map_gradient_inf_norm": record.map_gradient_inf_norm,
                    "map_newton_decrement": record.map_newton_decrement,
                    "chi_squared": record.chi_squared,
                    "n_degrees_of_freedom": record.n_degrees_of_freedom,
                    "map_time_seconds": record.map_time_seconds,
                    "laplace_time_seconds": record.laplace_time_seconds,
                    "mcmc_time_seconds": record.mcmc_time_seconds,
                    "total_time_seconds": record.total_time_seconds,
                }
            )
    return {
        "study_name": STUDY_NAME,
        "profile_family": PROFILE_FAMILY,
        "thomson_noise_model": NOISE_MODEL,
        "arms": {name: arm._asdict() for name, arm in ARMS.items()},
        "nuts": NUTS_CONFIGURATION.model_dump(mode="json"),
        "thinning": 1,
        "band_percentiles": [5.0, 95.0],
        "convergence_criteria": {
            "r_hat_max": mcmc_convergence_utils.R_HAT_LIMIT,
            "ess_bulk_min": mcmc_convergence_utils.ESS_BULK_LIMIT,
            "ess_tail_min": mcmc_convergence_utils.ESS_TAIL_LIMIT,
            "n_divergences": 0,
            "max_tree_depth_fraction": (
                mcmc_convergence_utils.MAX_TREE_DEPTH_FRACTION_LIMIT
            ),
        },
        "timing": (
            "wall clock in seconds, JAX compilation included; the first call of a run "
            "compiles, and every later one does not"
        ),
        "runs": runs,
    }


def stored_runs() -> dict[
    tuple[str, str, int], inference_types.PlasmaStateInferenceResult
]:
    """Every run in the manifest, keyed by truth, arm and noise seed.

    Returns:
        The records, each at the seed read off it.
    """
    runs = {}
    for label, identifier in sweep_utils.current_ids(IDS_FILE).items():
        record = inference_store_utils.load_record(
            inference_store_utils.INFERENCE_STORE, identifier
        )
        key = (
            record.truth_name,
            arm_of(label, record.truth_name),
            record.settings.seed,
        )
        runs[key] = record
    return runs


def arm_of(label: str, truth_name: str) -> str:
    """The arm a run label names, without its `_seed<seed>` suffix."""
    return re.sub(r"_seed\d+$", "", label.removeprefix(f"{truth_name}_"))


def cell_seeds(base_seed: int, truth_name: str, n_seeds: int) -> list[int]:
    """The noise seeds of one truth's draws, disjoint from every other truth's.

    Args:
        base_seed: The study's `--seed`.
        truth_name: Which truth the draws are for.
        n_seeds: How many draws that truth gets.

    Returns:
        `n_seeds` consecutive seeds, offset by the truth's catalog position.
    """
    offset = base_seed + SEED_STRIDE * sorted(desc_truth_types.DESC_TRUTH_IDS).index(
        truth_name
    )
    return [offset + replicate for replicate in range(n_seeds)]


def _settings(
    arguments: argparse.Namespace,
    arm: Arm,
    seed: int,
) -> inference_types.PlasmaEstimationSettings:
    """The inference configuration of one cell; only the arm varies.

    Every arm reports a band of as many draws as the sampler retains, so their
    percentiles carry the same Monte Carlo noise.
    """
    mcmc = NUTS_CONFIGURATION.model_copy(
        update={
            "n_warmup": arguments.n_warmup,
            "n_samples": arguments.n_samples,
            "n_chains": arguments.n_chains,
            "target_accept_prob": arguments.target_accept_prob,
            "init_strategy": arm.init_strategy,
        }
    )
    return inference_types.PlasmaEstimationSettings(
        method=arm.method,
        profile_family=PROFILE_FAMILY,
        seed=seed,
        add_measurement_noise=True,
        verbose=False,
        n_posterior_samples=mcmc.n_chains * mcmc.n_samples,
        mcmc=mcmc,
    )


def _parse_arguments() -> argparse.Namespace:
    """The study's command line; the sampler's budget defaults to the block above."""
    defaults = inference_types.PlasmaEstimationSettings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--truths",
        nargs="+",
        default=sorted(desc_truth_types.DESC_TRUTH_IDS),
        choices=sorted(desc_truth_types.DESC_TRUTH_IDS),
        help="Ground truths every recipe is run on.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        choices=list(ARMS),
        help="Arms to run; both by default.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=NUTS_CONFIGURATION.n_warmup,
        help="NUTS warmup iterations per chain, during which the step size and the "
        "mass matrix adapt.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=NUTS_CONFIGURATION.n_samples,
        help="NUTS draws retained per chain, none of them thinned.",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=NUTS_CONFIGURATION.n_chains,
        help="NUTS chains; they run one after another on a single CPU device.",
    )
    parser.add_argument(
        "--target-accept-prob",
        type=float,
        default=NUTS_CONFIGURATION.target_accept_prob,
        help="Acceptance probability the NUTS step size adapts towards; raising it "
        "shortens the step, the standard response to divergent transitions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Base seed; each truth's draws are `cell_seeds` of it.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Noisy draws per truth for map_laplace; NUTS runs the first only.",
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
