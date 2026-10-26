r"""What does solving for the equilibrium buy, against assuming one?

Every reconstruction on this layout has to say which flux surface a diagnostic point
sits on, and there are two ways to get there. The conventional one assumes an
equilibrium, inverts every point against it, and never revises it: the geometry is an
input. The one this folder builds keeps the surrogate in the loop, so the pressure the
profiles imply predicts the geometry those same profiles are then read on, and the two
move together.

This study runs both, on the same data, for each ground truth and each of `--n-seeds`
noise draws. The fixed arm is handed `FIXED_EQUILIBRIUM_ID` below and holds it for every
parameter value; the self-consistent arm is the ordinary run. Everything else is shared,
down to the seeds and the noise draws, so a difference between the columns of the table
is the geometry and nothing else. The tables average each arm over the draws.

The fixed equilibrium is deliberately **not** the truth. Handing the fixed arm the
equilibrium the data was generated from would measure nothing a real reconstruction can
have; a reference equilibrium from the training split is what is actually available
before the profiles are known, and the bias that follows from it being the wrong one is
the quantity of interest.

Both arms are `map_laplace`, a few minutes per cell:

    python run_study.py --truths wide          # one truth, both arms, to start
    python run_study.py --workers 4

Writes `profile_parameters.csv`, `profile_metrics.csv` and `inference_ids.json` beside
this script; the records
themselves go to `results_plots_tables/PlasmaStateInferenceResult/`.
"""

import argparse
import csv
import pathlib
import re
import sys

import numpy as np

_MODULE_DIRECTORY = pathlib.Path(__file__).resolve().parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parents[3]
sys.path.insert(0, str(_PACKAGE_DIRECTORY.parent))

from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E402,E501
    desc_truth_types,
    inference_types,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E402,E501
    diagnostic_design_point,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    inference_metrics_utils,
    inference_store_utils,
    parametrization_fit_utils,
    sweep_utils,
)

FIXED_EQUILIBRIUM_ID = desc_truth_types.REFERENCE_EQUILIBRIUM_ID
"""The one equilibrium the fixed arm assumes, for every truth and every parameter value.

A member of the **training** split, not of the test split the seven truths come from,
so no cell is handed its own answer; see `desc_truth_types.REFERENCE_EQUILIBRIUM_ID`
for how it was chosen. Hardcoded rather than exposed on the command line: it is the
study's definition, and a table whose fixed column came from a different geometry per
launch would not be one table.
"""

PROFILE_FAMILY = "two_power_zero_edge"
"""The family both arms infer in, `center * (1 - rho^alpha)^beta` per profile."""

ARMS: dict[str, bool] = {"fixed_equilibrium": True, "self_consistent": False}
"""The two arms, and whether each freezes the geometry at `FIXED_EQUILIBRIUM_ID` for
every parameter value, the conventional reconstruction. Also the CSV's column names."""

CSV_FILE = _MODULE_DIRECTORY / "profile_parameters.csv"
"""Target and inferred profile parameters, one row per parameter per truth."""

METRICS_FILE = _MODULE_DIRECTORY / "profile_metrics.csv"
"""Band half-width, RMSE and wall-clock time per truth, profile and arm.

The comparison in profile space, where the parameter table cannot be read: four of the
family's seven coefficients per profile are unconstrained by this layout, so a statistic
pooled over them reports the prior rather than the geometry.
"""

STUDY_NAME = _MODULE_DIRECTORY.name
"""What this study's records carry in `study_name`, and what its manifest selects on."""

DEFAULT_N_SEEDS = 1
"""Noisy draws per truth, which both arms share."""

SEED_STRIDE = 1000
"""Seeds between consecutive truths, so no two truths share a noise draw."""

IDS_FILE = _MODULE_DIRECTORY / "inference_ids.json"
"""One row per stored run, newest first, written by `sweep_utils.write_manifest`."""


def main() -> None:
    arguments = _parse_arguments()
    paths = sweep_utils.paths_from_arguments(arguments, STUDY_NAME)
    cells = [
        sweep_utils.SweepCell(
            name=f"{truth_name}_{arm_name}_seed{seed}",
            label=f"{truth_name:<16s} {arm_name} seed {seed}",
            truth_name=truth_name,
            settings=_settings(arguments, seed),
            design_point=diagnostic_design_point.DiagnosticDesignPoint(),
            fixed_equilibrium_id=FIXED_EQUILIBRIUM_ID if fixed_equilibrium else None,
        )
        for truth_name in arguments.truths
        for arm_name, fixed_equilibrium in ARMS.items()
        for seed in truth_seeds(arguments.seed, truth_name, arguments.n_seeds)
    ]
    sweep_utils.run_sweep(cells, paths, workers=arguments.workers)
    sweep_utils.write_manifest(IDS_FILE, STUDY_NAME)
    print(f"[study] manifest written to {IDS_FILE}")
    write_parameter_table(paths, arguments.truths, PROFILE_FAMILY)
    write_metrics_table()


def write_metrics_table() -> None:
    """Write the profile-space comparison of the arms off this study's stored records,
    each truth, arm and profile averaged over its noise draws.

    Reads the manifest rather than the sweep's return value, so it can be regenerated
    for a study that is already stored without running an inference again.
    """
    records = [
        inference_store_utils.load_record(inference_store_utils.INFERENCE_STORE, id_)
        for id_ in sweep_utils.current_ids(IDS_FILE).values()
    ]
    metrics = inference_metrics_utils.sweep_metrics_table(records)
    metrics["arm"] = [split_label(label)[1] for label in metrics["study_label"]]
    table = (
        metrics.groupby(["truth_name", "arm", "name"], sort=False)
        .agg(
            uncertainty_percent=("uncertainty_percent", "mean"),
            error_percent=("error_percent", "mean"),
            total_time_seconds=("total_time_seconds", "mean"),
            n_draws=("seed", "nunique"),
        )
        .reset_index()
        .rename(columns={"name": "profile"})
        .round({"uncertainty_percent": 4, "error_percent": 4, "total_time_seconds": 1})
    )
    table.to_csv(METRICS_FILE, index=False)
    print(f"[table] {len(table)} rows written to {METRICS_FILE}")


def split_label(study_label: str) -> tuple[str, str]:
    """The truth and the arm a cell's label is made of, with or without its seed."""
    stem = re.sub(r"_seed\d+$", "", study_label)
    for arm in sorted(ARMS, key=len, reverse=True):
        if stem.endswith(f"_{arm}"):
            return stem[: -len(arm) - 1], arm
    raise ValueError(f"{study_label!r} names no arm of this study")


def write_parameter_table(
    paths: sweep_utils.SweepPaths,
    truth_names: list[str],
    family: profile_parametrization_types.ProfileFamily,
) -> None:
    """Write the study's table: target parameters beside what each arm inferred.

    In the parameter space of the fitted family rather than in profile space, because
    that is where the two arms are directly comparable: a parameter has one target
    number per truth, so a bias shows as a column that sits off it, and the same table
    holds every truth. The target is the truth projected onto the family with
    `parametrization_fit_utils.target_parameters`, the truth itself being a 41-knot
    spline with no parameters of this family.

    The inferred values are each arm's MAP point averaged over the noise draws. The
    bands are not here: they are on the records, where the method that produced them
    put them, and the notebook reads them from there. A truth missing an arm is skipped
    rather than half-written, since a row with one arm missing invites a comparison
    against a blank.

    Args:
        paths: Where the records were stored.
        truth_names: The truths to write, in the order their rows appear.
        family: The parametrization every arm inferred in.
    """
    identifiers: dict[tuple[str, str], list[str]] = {}
    for label, identifier in sweep_utils.current_ids(IDS_FILE).items():
        identifiers.setdefault(split_label(label), []).append(identifier)
    parquet_path = paths.data_directory / f"{paths.dataset}.parquet"
    settings_path = paths.data_directory / f"{paths.dataset}_settings.json"
    table = []
    for truth_name in truth_names:
        if not all((truth_name, arm) in identifiers for arm in ARMS):
            print(f"[table] {truth_name}: incomplete pair, skipped")
            continue
        truth = desc_truth_types.read_desc_truth(
            truth_name, parquet_path, settings_path
        )
        targets = parametrization_fit_utils.target_parameters(family, truth.profiles)
        inferred = [
            tuple(
                np.mean(values, axis=0)
                for values in zip(
                    *(
                        _map_parameters(paths.store_directory, identifier)
                        for identifier in identifiers[(truth_name, arm)]
                    )
                )
            )
            for arm in ARMS
        ]
        for profile_index, profile in enumerate(
            ("electron_density", "electron_temperature")
        ):
            for parameter_index, target in enumerate(targets[profile_index]):
                table.append(
                    [
                        truth_name,
                        profile,
                        parameter_index,
                        f"{float(target):.6e}",
                        *(
                            f"{float(arm[profile_index][parameter_index]):.6e}"
                            for arm in inferred
                        ),
                    ]
                )

    with CSV_FILE.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth_name", "profile", "parameter_index", "target", *ARMS])
        writer.writerows(table)
    print(f"[table] {len(table)} rows written to {CSV_FILE}")


def truth_seeds(base_seed: int, truth_name: str, n_seeds: int) -> list[int]:
    """The noise seeds of one truth, which both arms share.

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


def _settings(
    arguments: argparse.Namespace, seed: int
) -> inference_types.PlasmaEstimationSettings:
    """The run configuration both arms share; only the geometry they read differs."""
    return inference_types.PlasmaEstimationSettings(
        method="map_laplace",
        profile_family=PROFILE_FAMILY,
        seed=seed,
        verbose=False,
    )


def _map_parameters(
    store_directory: pathlib.Path, inference_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """One record's MAP `ne` and `Te` parameter vectors, in the family's own units."""
    result, _ = inference_store_utils.load_inference(store_directory, inference_id)
    return (
        np.asarray(result.map_profiles.electron_density_params),
        np.asarray(result.map_profiles.electron_temperature_params),
    )


def _parse_arguments() -> argparse.Namespace:
    """The study's command line; the fixed equilibrium and the arms are fixed above."""
    defaults = inference_types.PlasmaEstimationSettings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--truths",
        nargs="+",
        default=sorted(desc_truth_types.DESC_TRUTH_IDS),
        choices=sorted(desc_truth_types.DESC_TRUTH_IDS),
        help="Ground truths to run every arm on; the default runs all seven.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Base seed; each truth's draws are `truth_seeds` of it.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Noisy draws per truth, shared by both arms.",
    )
    sweep_utils.add_sweep_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
