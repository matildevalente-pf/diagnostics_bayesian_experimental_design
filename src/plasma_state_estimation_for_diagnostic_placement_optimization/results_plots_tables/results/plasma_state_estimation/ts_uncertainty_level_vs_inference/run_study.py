r"""What the Thomson calibration uncertainty `N` costs the inferred profiles, against
how many interferometer chords the likelihood reads.

A sweep over the box `[1/N, N]` of the log-uniform prior on the per-volume calibration
corrections `delta_j` and over the chord configurations of `CHORD_FRACTIONS`, on every
catalog truth and `--n-seeds` noisy draws.

    python run_study.py --workers 5
    python run_study.py --levels 1 5 --chords 0 5 --workers 5
    python run_study.py --method mcmc_nuts --truths wide_1_parametric hollow_gp \
        --levels 1 1.2 5 --chords 5 --workers 6    # NUTS reference

Writes `inference_ids.json` and the comparison tables in `tables/` beside this script
(`inference_ids_mcmc_nuts.json` and no tables for `--method mcmc_nuts`), and one record
per run into the shared `results_plots_tables/PlasmaStateInferenceResult/`.
"""

import argparse
import math
import pathlib
import sys
import typing

import numpy as np

_MODULE_DIRECTORY = pathlib.Path(__file__).resolve().parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parents[3]
sys.path.insert(0, str(_PACKAGE_DIRECTORY.parent))

from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E402,E501
    desc_truth_types,
    inference_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E402,E501
    diagnostic_design_point,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_placement_optimization import (  # noqa: E402,E501
    diagnostics_placement_optimization_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    inference_metrics_utils,
    inference_store_utils,
    sweep_utils,
)

LEVELS = (1.0, 1.05, 1.2, 1.5, 2.0, 5.0, 10.0)
"""The values of `N` swept, the edge of the box `[1/N, N]` every `delta_j` keeps to.

One takes the geometric calibration as known and infers no `delta_j` at all, which is
the reference the others are read against; the rest ask what a calibration the design
pins down only to a factor `N` costs the profiles. The grid is dense where ln N is
comparable to the 5 % the data resolve each `delta_j` to, since with five chords the
posterior stops depending on `N` above about 5 (measured 2026-10-09 and 2026-10-10).
"""

CHORD_FRACTIONS: dict[int, tuple[float, ...]] = {
    0: (),
    1: (0.0,),
    2: (0.0, 1.0),
    3: (0.0, 0.5, 1.0),
    4: (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
    5: (0.0, 0.25, 0.5, 0.75, 1.0),
}
"""The chord configurations swept, by count: each chord's height as a fraction of the
pinned top chord's, from the central chord on the laser path (0) to the top one (1).

Upper half only, since at the design plane the cross-section is up-down symmetric and a
mirror chord would read the same line integral (author's decision, 2026-10-10).
"""

FAN_FRACTIONS = tuple(
    sorted(
        {fraction for fractions in CHORD_FRACTIONS.values() for fraction in fractions}
    )
)
"""The union of the configurations, the one fan every cell's placement carries.

A cell reads its configuration's chords and masks the rest, so the chord at a given
height carries the same noisy reading in every configuration and the Thomson signals
are the same in all of them.
"""

PINNED_FAN_CHORDS = 5
PINNED_FAN_MARGIN = 0.1
"""How `diagnostic_design_point.DEFAULT_CHORD_SURFACE_ANGLES` was built, five parallel
horizontal chords over 80 % of the plasma's vertical extent, which fixes the central and
top chords the fan interpolates between."""

METHOD: inference_types.InferenceMethod = "map_laplace"
"""The recipe every cell runs: the MAP with the Laplace approximation at it."""

NOISE_MODEL: thomson_scattering_utils.ThomsonNoiseModel = "electronic"
"""The Thomson uncertainty every cell carries, the W7-X detector noise curve
(`thomson_scattering_utils.signal_uncertainties`)."""

PROFILE_FAMILY = "two_power_zero_edge"
"""The family every cell infers in, `center * (1 - rho^alpha)^beta` per profile."""

DEFAULT_N_SEEDS = 1
"""Noisy draws per cell; the levels and configurations share their seeds, so the sweep
is paired."""

SEED_STRIDE = 1000
"""Seeds between consecutive truths, so no two truths ever share a draw."""

TABLE_VALUES = ("uncertainty_percent", "error_percent")
"""What each level's column block reports: band half-width and RMSE."""

STUDY_NAME = _MODULE_DIRECTORY.name
"""What this study's records carry in `study_name`, and what its manifest selects on."""

IDS_FILE = _MODULE_DIRECTORY / "inference_ids.json"
"""One row per stored run, written by `sweep_utils.write_manifest` off the records."""

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
"""The `study_name` of a `--method mcmc_nuts` record, so the sweep never loads it."""

NUTS_IDS_FILE = _MODULE_DIRECTORY / "inference_ids_mcmc_nuts.json"
"""The manifest of the `--method mcmc_nuts` runs, labelled as their Laplace twins."""

TABLES_DIRECTORY = _MODULE_DIRECTORY / "tables"
"""Where the comparison tables go, one CSV per profile and chord configuration."""


def main() -> None:
    arguments = _parse_arguments()
    sampled = arguments.method == "mcmc_nuts"
    study_name = NUTS_STUDY_NAME if sampled else STUDY_NAME
    ids_file = NUTS_IDS_FILE if sampled else IDS_FILE
    paths = sweep_utils.paths_from_arguments(arguments, study_name)
    stored = sweep_utils.current_ids(ids_file) if arguments.skip_stored else {}
    design_point = diagnostic_design_point.DiagnosticDesignPoint(
        thomson_noise_model=NOISE_MODEL,
        chord_surface_angles=fan_surface_angles(
            sweep_utils.reference_truth(
                paths.data_directory / f"{paths.dataset}.parquet",
                paths.data_directory / f"{paths.dataset}_settings.json",
            ),
            diagnostic_design_point.DiagnosticDesignPoint(),
        ),
    )
    cells = [
        sweep_utils.SweepCell(
            name=name,
            label=(f"{truth_name:<16s} N = {level:<5g} {n_chords} chords seed {seed}"),
            truth_name=truth_name,
            settings=_settings(arguments, seed, level),
            design_point=design_point,
            interferometer_chord_indices=chord_indices(n_chords),
        )
        for truth_name in arguments.truths
        for level in arguments.levels
        for n_chords in arguments.chords
        for seed in cell_seeds(arguments.seed, truth_name, arguments.n_seeds)
        if (name := cell_name(truth_name, level, n_chords, seed)) not in stored
    ]
    print(
        f"[sweep] {len(cells)} cells: {len(arguments.levels)} levels x "
        f"{len(arguments.chords)} chord configurations x {len(arguments.truths)} "
        f"truths x {arguments.n_seeds} draws in the {PROFILE_FAMILY} family, "
        f"{arguments.method}, Thomson noise {NOISE_MODEL}"
        + (f", {len(stored)} already stored and skipped" if stored else ""),
        flush=True,
    )
    sweep_utils.run_sweep(cells, paths, workers=arguments.workers)
    sweep_utils.write_manifest(ids_file, study_name)
    if not sampled:
        write_tables(arguments.levels, arguments.chords)


def write_tables(levels: list[float], chord_counts: list[int]) -> None:
    """Write one comparison table per profile and chord configuration off the study's
    stored records.

    Rows are the truths with a mean and a spread row below them, columns one block per
    level of `N` holding `TABLE_VALUES`.

    Args:
        levels: The levels whose column blocks the tables carry, in order.
        chord_counts: The chord configurations, one table each.
    """
    records = [
        inference_store_utils.load_record(inference_store_utils.INFERENCE_STORE, id_)
        for id_ in sweep_utils.current_ids(IDS_FILE).values()
    ]
    metrics = inference_metrics_utils.sweep_metrics_table(records)
    level_of = {
        record.inference_id: level_name(record.settings.thomson_calibration_bound)
        for record in records
    }
    metrics["arm"] = [level_of[identifier] for identifier in metrics["inference_id"]]
    TABLES_DIRECTORY.mkdir(exist_ok=True)
    for n_chords in chord_counts:
        selected = metrics.loc[metrics["n_interferometer_chords"] == n_chords]
        for profile in inference_metrics_utils.PROFILE_NAMES:
            table = inference_metrics_utils.arm_comparison_table(
                selected,
                profile,
                tuple(level_name(level) for level in levels),
                TABLE_VALUES,
            )
            path = TABLES_DIRECTORY / f"ts_calibration_bound_{profile}_if{n_chords}.csv"
            table.round(3).to_csv(path)
            print(f"[tables] {path}")


def fan_surface_angles(
    reference: desc_truth_types.DescTruth,
    design: diagnostic_design_point.DiagnosticDesignPoint,
) -> list[list[float]]:
    """The union fan as surface angles, its chords between the pinned central and top
    chords of the reference equilibrium at the heights `FAN_FRACTIONS`.

    Args:
        reference: The equilibrium every placement is built on.
        design: The design point supplying the plane.

    Returns:
        One `[start, end]` pair of surface angles per chord of the fan, in rad.
    """
    toroidal_angle = math.radians(design.toroidal_angle_deg)
    offsets, _, _, _ = diagnostics_placement_optimization_utils.parallel_chords(
        reference.coefficients,
        toroidal_angle,
        0.0,
        PINNED_FAN_CHORDS,
        PINNED_FAN_MARGIN,
    )
    central, top = offsets[PINNED_FAN_CHORDS // 2], offsets[-1]
    _, surface_angles, _, _ = (
        diagnostics_placement_optimization_utils.chords_at_offsets(
            reference.coefficients,
            toroidal_angle,
            0.0,
            central + np.asarray(FAN_FRACTIONS) * (top - central),
        )
    )
    return surface_angles.tolist()


def chord_indices(n_chords: int) -> tuple[int, ...]:
    """The fan indices of one chord configuration.

    Args:
        n_chords: Which configuration, by its count.

    Returns:
        The indices into `FAN_FRACTIONS` of the configuration's chords.
    """
    return tuple(
        FAN_FRACTIONS.index(fraction) for fraction in CHORD_FRACTIONS[n_chords]
    )


def cell_name(truth_name: str, level: float, n_chords: int, seed: int) -> str:
    """The label of one cell, the key of its manifest row and of its record.

    Args:
        truth_name: Which ground truth the cell reads.
        level: The value of `N`.
        n_chords: The chord configuration, by its count.
        seed: The noise seed.

    Returns:
        The name, e.g. `hollow_gp_n1p05_if3_seed42`.
    """
    return f"{truth_name}_{level_name(level)}_if{n_chords}_seed{seed}"


def level_name(level: float) -> str:
    """How one value of `N` is named in a label and a table column.

    Args:
        level: The value of `N`.

    Returns:
        The name, `n` and the value with `p` for the decimal point, `n1`, `n1p05`,
        `n1p2`, `n1p5`, `n2`, `n5`, `n10` for the default levels.
    """
    # The name is also a file stem, where a dot would read as a suffix and make the
    # figures of 1.05, 1.2 and 1.5 all overwrite those of 1.
    return f"n{level:g}".replace(".", "p")


def cell_seeds(base_seed: int, truth_name: str, n_seeds: int) -> list[int]:
    """The noise seeds of one truth's cells, disjoint from every other truth's.

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
    arguments: argparse.Namespace, seed: int, level: float
) -> inference_types.PlasmaEstimationSettings:
    """The inference configuration of one cell; the box edge and the seed vary.

    `delta_j,true = 1` in the data whatever `N` is, so an arm differs from its
    neighbours in the prior alone and two arms of one seed are fitted to identical data.
    """
    return inference_types.PlasmaEstimationSettings(
        method=arguments.method,
        mcmc=(
            NUTS_CONFIGURATION
            if arguments.method == "mcmc_nuts"
            else inference_types.McmcSettings()
        ),
        profile_family=PROFILE_FAMILY,
        thomson_calibration_bound=level,
        seed=seed,
        add_measurement_noise=True,
        verbose=False,
    )


def _parse_arguments() -> argparse.Namespace:
    """The study's command line; the swept quantities are the box edge `N` and the
    chord configuration."""
    defaults = inference_types.PlasmaEstimationSettings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--truths",
        nargs="+",
        default=sorted(desc_truth_types.DESC_TRUTH_IDS),
        choices=sorted(desc_truth_types.DESC_TRUTH_IDS),
        help="Ground truths every level is run on.",
    )
    parser.add_argument(
        "--method",
        default=METHOD,
        choices=list(typing.get_args(inference_types.InferenceMethod)),
        help=(
            "Recipe every cell runs; `mcmc_nuts` stores under its own study name and "
            "manifest and writes no tables."
        ),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=list(LEVELS),
        help="Values of `N` to sweep, each at least one.",
    )
    parser.add_argument(
        "--chords",
        nargs="+",
        type=int,
        default=sorted(CHORD_FRACTIONS),
        choices=sorted(CHORD_FRACTIONS),
        help="Chord configurations to sweep, by count.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Noisy draws per cell.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Base seed; each truth's draws are `cell_seeds` of it.",
    )
    parser.add_argument(
        "--skip-stored",
        action="store_true",
        help="Skip cells the manifest already has a current run for.",
    )
    sweep_utils.add_sweep_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
