# Running many inferences in parallel and recording which id came from which cell.
#
# Every study in this folder is the same shape: build a list of settings that differ in
# one or two fields, run them concurrently, store each result, and write a manifest the
# notebook reads so it never has to hardcode an id. That machinery lives here once
# rather than being copied into each study's runner.
#
# Cells run in separate processes, not threads. JAX does not release its memory between
# runs inside one process, so a long sweep in a single process grows until it is killed;
# `max_tasks_per_child=1` gives every cell a fresh interpreter. The consequence is that
# each cell reloads the surrogate, which costs a few seconds and is the price of the
# isolation.
#
# Memory, not cores, bounds the worker count: about 2.5 GB per JAX process.

import argparse
import concurrent.futures
import json
import os
import pathlib
import time
from typing import NamedTuple

import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    inference_run_utils,
    inference_types,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_utils,
    model_storage_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E501
    inference_store_utils,
    synthetic_target_utils,
)

DEFAULT_WORKERS = 4
"""Concurrent cells.

Four fits roughly 10 GB free against about 2.5 GB per process.
"""

_PACKAGE_DIRECTORY = pathlib.Path(__file__).resolve().parents[2]


class SweepPaths(NamedTuple):
    """Everything a worker needs to find its inputs and put its output."""

    data_directory: pathlib.Path
    """Holds the dataset parquet and its settings JSON."""

    dataset: str
    """Stem of those two files."""

    models_directory: pathlib.Path
    """Holds the stored surrogates."""

    model_name: str
    """Stem of the surrogate every cell uses as its forward model."""

    study_name: str = ""
    """Folder name of the study, written on every record so its manifest can select the
    study's runs out of the shared store."""

    store_directory: pathlib.Path = inference_store_utils.INFERENCE_STORE
    """Where the records go, one directory per inference id, shared by every study."""

    placements_directory: pathlib.Path = inference_store_utils.PLACEMENT_STORE
    """Where the placements go, one directory per placement id, shared by all."""


class SweepCell(NamedTuple):
    """One point of a study: a truth, a full settings object, and a name for the log."""

    name: str
    """Filename-safe identifier of this cell, e.g. `ts_10_if_05`.

    The key this cell gets in the manifest, the value of `study_label` on its record,
    and the name of any per-cell directory, so one string ties the three together.
    """

    label: str
    """Short human-readable description, printed as the cell completes."""

    truth_name: str
    """Which ground truth generates this cell's synthetic data."""

    settings: inference_types.PlasmaEstimationSettings
    """The inference configuration; whatever the study varies is already baked in."""

    design_point: diagnostic_design_point.DiagnosticDesignPoint
    """The instrument, including every noise magnitude.

    A study that sweeps the noise varies this rather than the settings.
    """

    fixed_equilibrium_id: str | None = None
    """Freeze the geometry at this equilibrium instead of predicting it per parameter.

    The cell then runs the conventional reconstruction, which assumes an equilibrium and
    inverts every diagnostic point against it, rather than the self-consistent one this
    folder does everywhere else. None keeps the surrogate in the loop.

    An id rather than a coefficient object because a cell crosses a process boundary and
    is meant to stay readable; the worker reads it from the same parquet the truth comes
    from, and it lands on the record as `fixed_equilibrium_id` so a stored run says
    which geometry it assumed. It need not be, and normally is not, the truth's own: a
    reference equilibrium from the training split is what a real reconstruction would
    have to hand.
    """

    interferometer_chord_indices: tuple[int, ...] | None = None
    """Keep exactly these chords of the fan, masking the rest; None keeps every chord.

    An empty tuple masks the whole diagnostic, which is not the same as removing it: the
    geometry stays on the record, so the layout figure can still show what was switched
    off.
    """

    warm_start: dict[str, float] | None = None
    """MAP point to start from once every prior draw has failed, one value per sample
    site; None tries the prior draws only.

    See `inference_run_utils.run_inference`.
    """


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the surrogate, worker and path options every sweep runner shares.

    Args:
        parser: The runner's parser; the options land on it in place.
    """
    parser.add_argument(
        "--model-name",
        default=inference_types.DEFAULT_SURROGATE_NAME,
        help="Stem of the stored surrogate every cell uses as its forward model.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent cells; about 2.5 GB of memory each.",
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=_PACKAGE_DIRECTORY
        / "mhd_equilibrium_forward_models"
        / "trained_models",
        help="Directory holding the stored surrogates.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_PACKAGE_DIRECTORY.parents[1] / "data",
        help="Directory holding the dataset parquet and its settings JSON.",
    )
    parser.add_argument(
        "--dataset",
        default=dataset_utils.DEFAULT_DATASET,
        help="Stem of the dataset files.",
    )


def paths_from_arguments(arguments: argparse.Namespace, study_name: str) -> SweepPaths:
    """Assemble a `SweepPaths` from the options `add_sweep_arguments` defined.

    Args:
        arguments: The parsed command line.
        study_name: Folder name of the study launching the sweep.

    Returns:
        The paths every cell of the study reads and writes.
    """
    return SweepPaths(
        data_directory=arguments.data_dir,
        dataset=arguments.dataset,
        models_directory=arguments.models_dir,
        model_name=arguments.model_name,
        study_name=study_name,
    )


def run_sweep(
    cells: list[SweepCell],
    paths: SweepPaths,
    workers: int = DEFAULT_WORKERS,
) -> list[dict]:
    """Run every cell concurrently and return one manifest row each.

    A cell that raises is recorded with a null `inference_id` and its error rather than
    stopping the sweep, because losing the other cells to one bad configuration wastes
    far more than it protects.

    Args:
        cells: The points to run.
        paths: Shared input and output locations.
        workers: How many cells run at once.

    Returns:
        One row per cell, in completion order.
    """
    paths.store_directory.mkdir(parents=True, exist_ok=True)
    print(
        f"[sweep] {len(cells)} cells, {workers} workers, "
        f"storing in {paths.store_directory}",
        flush=True,
    )
    started = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, max_tasks_per_child=1
    ) as pool:
        futures = {pool.submit(run_cell, cell, paths, workers): cell for cell in cells}
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            cell = futures[future]
            try:
                rows.append(future.result())
                status = rows[-1]["inference_id"]
            except Exception as error:
                status = f"FAILED {type(error).__name__}: {error}"
                rows.append(
                    {
                        "study_label": cell.name,
                        "inference_id": None,
                        "error": str(error),
                    }
                )
            print(f"[sweep] {done}/{len(cells)}  {cell.label}  {status}", flush=True)

    failed = [row for row in rows if row["inference_id"] is None]
    print(
        f"\n[sweep] {len(rows) - len(failed)}/{len(rows)} succeeded in "
        f"{(time.perf_counter() - started) / 60:.1f} min"
    )
    if failed:
        print(f"[sweep] {len(failed)} cells FAILED and are absent from the manifest")
    return rows


def run_cell(cell: SweepCell, paths: SweepPaths, workers: int) -> dict:
    """Run one cell to completion, store it, and summarize it for the manifest.

    Runs in its own process, so it loads the surrogate itself.

    Args:
        cell: What to run.
        paths: Where its inputs and outputs live.
        workers: Used only to divide the cores between concurrent cells.

    Returns:
        The cell's name and the id of the record it stored.
    """
    # Without this each of the concurrent processes spawns a thread pool as wide as the
    # whole machine, and they spend their time contending rather than computing.
    threads = str(max(1, (os.cpu_count() or 8) // max(workers, 1)))
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={threads}",
    )

    settings = cell.settings
    parquet_path = paths.data_directory / f"{paths.dataset}.parquet"
    settings_path = paths.data_directory / f"{paths.dataset}_settings.json"
    truth = desc_truth_types.read_desc_truth(
        cell.truth_name, parquet_path, settings_path
    )
    surrogate = model_storage_utils.load_surrogate(
        paths.models_directory,
        paths.model_name,
        paths.data_directory,
        machine_scale=truth.machine_scale,
    )
    placement = synthetic_target_utils.build_diagnostic_placement(
        cell.design_point, reference_truth(parquet_path, settings_path)
    )
    placement_id = inference_store_utils.save_placement(
        paths.placements_directory, placement
    )
    target = synthetic_target_utils.build_synthetic_target(
        truth,
        placement,
        interferometer_n_sample=settings.interferometer_n_sample,
        trim_chords_to_plasma=settings.trim_chords_to_plasma,
        seed=settings.seed,
        add_noise=settings.add_measurement_noise,
        interferometer_sigma=settings.interferometer_sigma,
    )
    chord_indices = (
        None
        if cell.interferometer_chord_indices is None
        else tuple(sorted(int(index) for index in cell.interferometer_chord_indices))
    )
    target = synthetic_target_utils.select_chords(target, chord_indices)
    fixed_coefficients = (
        desc_truth_types.read_desc_truth(
            cell.fixed_equilibrium_id, parquet_path, settings_path
        ).coefficients
        if cell.fixed_equilibrium_id is not None
        else None
    )
    # After the masking, so a volume the likelihood does not read, whose sigma is zero,
    # gets no calibration correction of its own.
    thomson_target = target.thomson_signals
    infer_calibration = settings.thomson_calibration_bound > 1.0
    active_volumes = (
        np.flatnonzero(np.any(np.asarray(thomson_target.uncertainties) > 0.0, axis=1))
        if infer_calibration and thomson_target is not None
        else []
    )
    priors = priors_utils.default_priors(
        settings.profile_family,
        calibration_volume_indices=tuple(int(index) for index in active_volumes),
        calibration_bound=settings.thomson_calibration_bound,
    )
    result, arrays = inference_run_utils.run_inference(
        settings,
        priors,
        target,
        surrogate,
        truth,
        design_point=cell.design_point,
        fixed_coefficients=fixed_coefficients,
        warm_start=cell.warm_start,
    )
    # The study provenance goes onto the record, not into the manifest: a record that
    # describes itself cannot fall out of step with a file that points at it.
    result = result.model_copy(
        update={
            "study_name": paths.study_name,
            "study_label": cell.name,
            "dataset": paths.dataset,
            "interferometer_chord_indices": (
                None if chord_indices is None else list(chord_indices)
            ),
            "fixed_equilibrium_id": cell.fixed_equilibrium_id or "",
            "diagnostics_placement_id": placement_id,
        }
    )
    inference_store_utils.save_inference(paths.store_directory, result, arrays)
    return {"study_label": cell.name, "inference_id": result.inference_id}


def reference_truth(
    parquet_path: pathlib.Path, settings_path: pathlib.Path
) -> desc_truth_types.DescTruth:
    """The reference equilibrium every placement is built on.

    Args:
        parquet_path: Path of the dataset parquet.
        settings_path: Path of its settings JSON.

    Returns:
        The truth at `desc_truth_types.REFERENCE_EQUILIBRIUM_ID`.
    """
    return desc_truth_types.read_desc_truth(
        desc_truth_types.REFERENCE_EQUILIBRIUM_ID, parquet_path, settings_path
    )


def write_manifest(path: pathlib.Path, study_name: str) -> None:
    """Write one row per stored run of a study, newest first, off the records.

    A row carries when the run finished, its id, and the three things that say what it
    was: which truth, which family, which method. `current` marks the newest row of each
    `study_label`, which is the run a notebook loads for that cell and what
    `current_ids` reads.

    Derived from the store rather than assembled from whichever launch happens to be
    finishing. A launch can therefore no longer drop a row it did not produce, which is
    what the merge this function used to need existed to prevent and what a failed cell
    defeated anyway; and a run superseded by a later one stays in the file, marked, in
    place of vanishing with the entry that named it.

    Nothing here is a second copy of a result. Every field is an identifier read back
    off the record that owns it, so the file cannot disagree with the store, only be out
    of date, and any launch brings it back. How well a run recovered the truth, what it
    was configured with and what it cost stay on the record, where the run that knows
    them wrote them.

    Args:
        path: Where to write the manifest.
        study_name: Folder name of the study, as its records carry in `study_name`.
    """
    summaries = [
        summary
        for summary in inference_store_utils.summarize_store(
            inference_store_utils.INFERENCE_STORE
        )
        if summary["study_name"] == study_name
    ]
    newest: dict[str, str] = {}
    for summary in summaries:
        if summary["study_label"]:
            newest.setdefault(summary["study_label"], summary["inference_id"])
    path.write_text(
        json.dumps(
            [
                {
                    "timestamp": summary["generated"],
                    "inference_id": summary["inference_id"],
                    "profile_name": summary["truth_name"],
                    "family": summary["profile_family"],
                    "method": summary["method"],
                    "study_label": summary["study_label"],
                    "current": newest.get(summary["study_label"])
                    == summary["inference_id"],
                }
                for summary in summaries
            ],
            indent=2,
        )
        + "\n"
    )


def current_ids(path: pathlib.Path) -> dict[str, str]:
    """The id a notebook loads for each cell, read off the manifest, naming every run in
    which no MAP start converged; such a run is kept, its band the best unconverged
    start's Gaussian.

    Args:
        path: A manifest written by `write_manifest`.

    Returns:
        Study label to inference id, one entry per label and the newest run of it;
        empty when the manifest has not been written yet.
    """
    if not path.exists():
        return {}
    ids = {
        row["study_label"]: row["inference_id"]
        for row in json.loads(path.read_text())
        if row["current"]
    }
    # Read off the record's JSON alone, so the check costs no array load per run.
    for label, identifier in ids.items():
        record = json.loads(
            (
                inference_store_utils.INFERENCE_STORE
                / identifier
                / inference_store_utils.RECORD_FILE
            ).read_text()
        )
        if record.get("n_map_starts_converged", 1) == 0:
            print(
                f"[manifest] {label} ({identifier}): no MAP start converged; its band "
                "is the best unconverged start's Gaussian"
            )
    return ids
