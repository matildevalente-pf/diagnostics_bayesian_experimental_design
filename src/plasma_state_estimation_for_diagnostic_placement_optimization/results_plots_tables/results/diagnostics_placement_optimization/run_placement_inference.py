"""`map_laplace` or NUTS on the layouts a placement study compares, on every plasma
state of the catalogue, in parallel; the study's notebook reads the runs off the
manifest this writes beside it.

    python run_placement_inference.py --study if_placement_study --workers 4
    python run_placement_inference.py --study if_placement_study \
        --scan if_placement_study/tables/if_placement_scan_chord_counts.csv \
        --layouts 3_chords_optimized 4_chords_optimized 5_chords_optimized \
        6_chords_optimized --workers 4
    python run_placement_inference.py --study if_and_ts_placement_study --workers 4
    python run_placement_inference.py --study if_and_ts_placement_study \
        --method mcmc_nuts --layouts optimized --workers 4
    python run_placement_inference.py --study if_and_ts_placement_study \
        --layouts optimized --truths wide_2_parametric --warm-start-from-benchmark
    python run_placement_inference.py --study if_and_ts_placement_study \
        --layouts separately_optimized --workers 4
    python run_placement_inference.py --study if_and_ts_placement_study \
        --layouts initial_guess optimized --truths test_state_1_parametric \
        test_state_2_parametric test_state_3_parametric test_state_4_parametric \
        test_state_5_gp test_state_6_gp test_state_7_gp test_state_8_gp --workers 4
"""

import argparse
import math
import pathlib
import sys
import types
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

_MODULE_DIRECTORY = pathlib.Path(__file__).resolve().parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parents[2]
sys.path.insert(0, str(_PACKAGE_DIRECTORY.parent))

from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E402,E501
    desc_truth_types,
    inference_types,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E402,E501
    diagnostic_design_point,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_placement_optimization import (  # noqa: E402,E501
    diagnostics_placement_optimization_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.results.diagnostics_placement_optimization.if_and_ts_placement_study import (  # noqa: E402,E501
    run_scan,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.results.diagnostics_placement_optimization.if_placement_study import (  # noqa: E402,E501
    run_study,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.results.plasma_state_estimation.mcmc_benchmark import (  # noqa: E402,E501
    run_study as mcmc_benchmark,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    inference_metrics_utils,
    inference_store_utils,
    sweep_utils,
)


class Layout(NamedTuple):
    """One diagnostic configuration a study compares, under the name its runs carry."""

    name: str
    """Filename-safe label, part of every run's `study_label`."""

    design_point: diagnostic_design_point.DiagnosticDesignPoint
    """The instrument."""


STATE_NAMES = (
    *sorted(desc_truth_types.DESC_TRUTH_IDS),
    *desc_truth_types.TEST_STATE_IDS,
)
"""Every plasma state a run may be on, in the order that sets its seed: the catalogue,
then the test states no layout was optimized on."""

STUDIES: dict[str, types.ModuleType] = {
    "if_placement_study": run_study,
    "if_and_ts_placement_study": run_scan,
}
"""The studies this runs for, each a module with `OUTPUT`, `compared_layouts` and
`design_point`, keyed by the folder its records carry in `study_name`."""

METRICS = ("error_percent", "uncertainty_percent")
"""The `profile_accuracy` fields the tables report: RMSE of the reported mean in percent
of the truth's peak, and median half-width of the 5 to 95 percent band in percent."""

PROFILES = ("electron_density", "electron_temperature")
"""The profiles the tables report, in this order."""

SEED_STRIDE = 1000
"""Seeds between consecutive plasma states, so no two states share a draw."""

DEFAULT_N_SEEDS = 1
"""Noisy draws per case; draw `k` of a state is the same for every layout."""


def run_cases(
    cases: Sequence[tuple[Layout, str]],
    method: inference_types.InferenceMethod,
    paths: sweep_utils.SweepPaths,
    ids_file: pathlib.Path,
    workers: int = sweep_utils.DEFAULT_WORKERS,
    seed: int = inference_types.PlasmaEstimationSettings().seed,
    n_seeds: int = DEFAULT_N_SEEDS,
    skip_stored: bool = False,
    warm_start_from_benchmark: bool = False,
) -> dict[str, str]:
    """Run `method` on every (layout, plasma state) case in parallel and store the runs.

    Args:
        cases: The layout and the name of the plasma state of each run.
        method: The inference recipe every run uses.
        paths: Where the inputs live and the records go; `study_name` tags the records.
        ids_file: Manifest written beside the study and read back by its notebook.
        workers: Concurrent runs, about 2.5 GB of memory each.
        seed: Base seed; draw `k` of catalogue state `i` reads `seed + 1000 i + k`.
        n_seeds: Noisy draws per case, shared by every layout of a state.
        skip_stored: Leave out the cases the manifest already holds a run of.
        warm_start_from_benchmark: Start every run at `benchmark_warm_start`, before
            any prior draw, to restart a run that ended in a local minimum.

    Returns:
        Study label to inference id of the current run of every case.
    """
    stored = sweep_utils.current_ids(ids_file) if skip_stored else {}
    cells = [
        cell
        for layout, truth_name in cases
        for draw in range(n_seeds)
        if (
            cell := _cell(
                layout, truth_name, draw, method, seed, warm_start_from_benchmark
            )
        ).name
        not in stored
    ]
    print(
        f"[placement] {len(cells)} runs of {method}, {len(stored)} already stored",
        flush=True,
    )
    if cells:
        sweep_utils.run_sweep(cells, paths, workers=workers)
    sweep_utils.write_manifest(ids_file, paths.study_name)
    return sweep_utils.current_ids(ids_file)


def run_label(
    layout_name: str,
    truth_name: str,
    method: inference_types.InferenceMethod = "map_laplace",
    draw: int = 0,
) -> str:
    """The `study_label` of one run, the key its notebook reads it by."""
    suffix = "_nuts" if method == "mcmc_nuts" else ""
    return f"{truth_name}_{layout_name}{suffix}_seed{draw:02d}"


def study_layouts(
    study: types.ModuleType,
    paths: sweep_utils.SweepPaths,
    scan: pathlib.Path | None = None,
) -> list[Layout]:
    """The layouts a study compares, read off its scan.

    Args:
        study: A value of `STUDIES`.
        paths: Where the dataset and the surrogate live, to score the initial guess.
        scan: The scan table to read them from; the study's `OUTPUT` if None.

    Returns:
        One `Layout` per compared row, in the study's order.
    """
    rows = study.compared_layouts(pd.read_csv(scan or study.OUTPUT), paths)
    return [Layout(name, study.design_point(row)) for name, row in rows.items()]


def ids_file(study_name: str) -> pathlib.Path:
    """The manifest of a study, beside its notebook."""
    return _MODULE_DIRECTORY / study_name / "inference_ids.json"


def benchmark_warm_start(
    truth_name: str, seed: int, n_volumes: int
) -> dict[str, float]:
    """The MAP of the same plasma state's `map_laplace` run in the MCMC benchmark, with
    every calibration correction at its nominal value, as the start of a run here.

    Args:
        truth_name: The plasma state.
        seed: The benchmark run's seed, which the run restarted from it shares.
        n_volumes: Scattering volumes of the layout, one calibration site each.

    Returns:
        One value per sample site of the run.
    """
    record = inference_store_utils.load_record(
        inference_store_utils.INFERENCE_STORE,
        sweep_utils.current_ids(mcmc_benchmark.IDS_FILE)[
            f"{truth_name}_map_laplace_seed{seed}"
        ],
    )
    return {
        **{
            f"ne_{index}": float(value)
            for index, value in enumerate(record.map_profiles.electron_density_params)
        },
        **{
            f"te_{index}": float(value)
            for index, value in enumerate(
                record.map_profiles.electron_temperature_params
            )
        },
        **{
            f"{priors_utils.CALIBRATION_PREFIX}_{index}": 1.0
            for index in range(n_volumes)
        },
    }


def accuracy_table(
    run_ids: Mapping[str, str],
    layout_names: Sequence[str],
    truth_names: Sequence[str] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
    method: inference_types.InferenceMethod = "map_laplace",
) -> pd.DataFrame:
    """RMSE and band half-width of `ne` and `Te` of every stored run of the layouts.

    Args:
        run_ids: A manifest as `sweep_utils.current_ids` reads it.
        layout_names: The layouts to report, in the order the table lists them.
        truth_names: The plasma states each layout was run on.
        method: The recipe of the runs.

    Returns:
        One row per layout, plasma state, noisy draw the manifest holds and profile of
        `PROFILES`, with the `METRICS`, whether the MAP converged and the information
        gain of the Laplace Gaussian at the MAP.
    """
    rows = []
    for layout in layout_names:
        for truth_name in truth_names:
            # Draws are stored from 0 up, so the first missing one ends the case.
            draw = 0
            while (label := run_label(layout, truth_name, method, draw)) in run_ids:
                record = inference_store_utils.load_record(
                    inference_store_utils.INFERENCE_STORE, run_ids[label]
                )
                for accuracy in inference_metrics_utils.profile_accuracy(record):
                    if accuracy.name in PROFILES:
                        rows.append(
                            {
                                "layout": layout,
                                "truth_name": truth_name,
                                "draw": draw,
                                "profile": accuracy.name,
                                "map_converged": record.map_converged,
                                "information_gain_bits": record.information_gain_bits,
                                **{m: getattr(accuracy, m) for m in METRICS},
                            }
                        )
                draw += 1
            if draw == 0:
                raise KeyError(
                    f"no stored run of {run_label(layout, truth_name, method)}"
                )
    return pd.DataFrame(rows)


def state_means(per_run: pd.DataFrame) -> pd.DataFrame:
    """Each plasma state's mean over its noisy draws, so that a standard error over the
    states counts every state once whatever its number of draws."""
    return pd.DataFrame(
        per_run.groupby(
            ["layout", "truth_name", "profile"], sort=False, as_index=False
        ).agg(
            n_draws=("draw", "size"),
            map_converged=("map_converged", "all"),
            information_gain_bits=("information_gain_bits", "mean"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
    )


def accuracy_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard error over the plasma states of each layout's `METRICS`.

    Args:
        per_run: The table `accuracy_table` returns.

    Returns:
        One row per layout; one column per profile and metric, `mean +- standard
        error` of the plasma states' means over their draws.
    """
    per_run = state_means(per_run)
    layouts = list(dict.fromkeys(per_run["layout"]))
    columns: dict[tuple[str, str], list[str]] = {}
    for profile in PROFILES:
        for metric in METRICS:
            columns[(profile, metric)] = []
            for layout in layouts:
                values = np.asarray(
                    per_run.loc[
                        (per_run["layout"] == layout) & (per_run["profile"] == profile),
                        metric,
                    ],
                    dtype=float,
                )
                standard_error = values.std(ddof=1) / math.sqrt(values.size)
                columns[(profile, metric)].append(
                    f"{values.mean():.2f} +- {standard_error:.2f}"
                )
    return pd.DataFrame(columns, index=pd.Index(layouts, name="layout"))


def _cell(
    layout: Layout,
    truth_name: str,
    draw: int,
    method: inference_types.InferenceMethod,
    base_seed: int,
    warm_start_from_benchmark: bool = False,
) -> sweep_utils.SweepCell:
    """One run of `method` on noisy draw `draw` of a plasma state and a layout; a NUTS
    run shares the seed, hence the data, of the `map_laplace` run of the same draw."""
    settings = inference_types.PlasmaEstimationSettings(
        method=method,
        profile_family="two_power_zero_edge",
        seed=base_seed + SEED_STRIDE * STATE_NAMES.index(truth_name) + draw,
        verbose=False,
    )
    if method == "mcmc_nuts":
        mcmc = mcmc_benchmark.NUTS_CONFIGURATION
        settings = settings.model_copy(
            update={"mcmc": mcmc, "n_posterior_samples": mcmc.n_chains * mcmc.n_samples}
        )
    warm_start = None
    if warm_start_from_benchmark:
        settings = settings.model_copy(
            update={"map": settings.map.model_copy(update={"warm_start_first": True})}
        )
        warm_start = benchmark_warm_start(
            truth_name, settings.seed, len(layout.design_point.volume_span_fractions)
        )
    return sweep_utils.SweepCell(
        name=run_label(layout.name, truth_name, method, draw),
        label=f"{truth_name:<20s} {layout.name} draw {draw} {method}",
        truth_name=truth_name,
        settings=settings,
        design_point=layout.design_point,
        warm_start=warm_start,
    )


def _parse_arguments() -> argparse.Namespace:
    """The command line; the layouts come from the study's scan."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--study",
        required=True,
        choices=list(STUDIES),
        help="Whose compared layouts to run, and where the manifest goes.",
    )
    parser.add_argument(
        "--method",
        default="map_laplace",
        choices=("map_laplace", "mcmc_nuts"),
        help="Recipe every run uses.",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=None,
        help="Run only these of the study's compared layouts, by name; all by default.",
    )
    parser.add_argument(
        "--scan",
        type=pathlib.Path,
        default=None,
        help="Scan table to read the layouts from instead of the study's own, such as "
        "the chord-count scan of if_placement_study.",
    )
    parser.add_argument(
        "--truths",
        nargs="+",
        default=list(diagnostics_placement_optimization_utils.TRUTH_NAMES),
        choices=STATE_NAMES,
        help="Plasma states to run every layout on; the whole catalogue by default, "
        "or the test states no layout was optimized on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=inference_types.PlasmaEstimationSettings().seed,
        help="Base seed; draw `k` of catalogue state `i` reads `seed + 1000 i + k`.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Noisy draws per case, shared by every layout of a state.",
    )
    parser.add_argument(
        "--skip-stored",
        action="store_true",
        help="Skip the cases the manifest already holds a run of, which resumes an "
        "interrupted launch instead of storing a second run of every case.",
    )
    parser.add_argument(
        "--warm-start-from-benchmark",
        action="store_true",
        help="Start every run at the MAP of the same plasma state's map_laplace run in "
        "the MCMC benchmark, calibrations nominal, before any prior draw; restarts "
        "runs that ended in a local minimum.",
    )
    sweep_utils.add_sweep_arguments(parser)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    paths = sweep_utils.paths_from_arguments(arguments, arguments.study)
    layouts = study_layouts(STUDIES[arguments.study], paths, arguments.scan)
    if arguments.layouts:
        unknown = set(arguments.layouts) - {layout.name for layout in layouts}
        if unknown:
            raise SystemExit(
                f"unknown layouts {sorted(unknown)}; the study compares "
                f"{[layout.name for layout in layouts]}"
            )
        layouts = [layout for layout in layouts if layout.name in arguments.layouts]
    cases = [
        (layout, truth_name) for layout in layouts for truth_name in arguments.truths
    ]
    manifest = ids_file(arguments.study)
    run_cases(
        cases,
        arguments.method,
        paths,
        manifest,
        workers=arguments.workers,
        seed=arguments.seed,
        n_seeds=arguments.n_seeds,
        skip_stored=arguments.skip_stored,
        warm_start_from_benchmark=arguments.warm_start_from_benchmark,
    )
    print(f"[placement] manifest written to {manifest}")


if __name__ == "__main__":
    main()
