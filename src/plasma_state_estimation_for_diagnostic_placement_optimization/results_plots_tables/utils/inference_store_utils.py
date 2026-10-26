# Storing one inference run under a short random id, and reading it back.
#
# Every study's runs live in `results_plots_tables/PlasmaStateInferenceResult/` and every
# placement in `results_plots_tables/DiagnosticPlacement/`, each folder named after the
# object it holds; a study's own folder keeps only the id manifest.
#
# A run writes `<directory>/<id>/record.json` and `<directory>/<id>/arrays.npz`: the
# record is everything a table needs and is readable without this package, the npz is
# everything a figure needs. Same split as `model_storage_utils` uses for a trained
# surrogate, and for the same reason.
#
# The id is what makes a notebook reproducible. Naming the id at the top of a notebook
# pins every figure below it to one run, so a figure cannot silently come from whatever
# was computed last. It is random rather than a timestamp or a description because it
# has to be short enough to paste and stable enough to cite; the timestamp, the method
# and the configuration all live inside the record, where they can be read but not
# mistyped. Eight hex characters is short enough to paste and, at a few hundred runs,
# carries a collision chance below one in ten thousand; `save_inference` refuses to
# overwrite anyway, so a collision is loud rather than silent.

import json
import pathlib

import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
)

STORE_ROOT = pathlib.Path(__file__).resolve().parents[1]

INFERENCE_STORE = STORE_ROOT / "PlasmaStateInferenceResult"
"""Every stored run of every study, one directory per inference id."""

PLACEMENT_STORE = STORE_ROOT / "DiagnosticPlacement"
"""Every stored placement, one directory per placement id."""

RECORD_FILE = "record.json"
"""Name of the JSON side of a stored run."""

ARRAYS_FILE = "arrays.npz"
"""Name of the array side of a stored run."""

PLACEMENT_FILE = "placement.json"
"""Name of a stored diagnostic placement, under `<directory>/<placement_id>/`."""


def save_inference(
    directory: pathlib.Path,
    result: inference_types.PlasmaStateInferenceResult,
    arrays: dict[str, np.ndarray],
) -> pathlib.Path:
    """Write one run's record and arrays under its own id.

    Args:
        directory: Root of the store; the run gets a subdirectory named by its id.
        result: The record, whose `inference_id` names the subdirectory.
        arrays: Everything a figure needs, keyed by name.

    Returns:
        The directory the run was written to.

    Raises:
        FileExistsError: If that id is already stored, which means either a collision or
            a caller reusing an id, and overwriting either would lose a run.
    """
    run_directory = directory / result.inference_id
    if run_directory.exists():
        raise FileExistsError(
            f"{run_directory} already exists; inference id "
            f"{result.inference_id!r} would be overwritten. Generate a new id."
        )
    run_directory.mkdir(parents=True)
    (run_directory / RECORD_FILE).write_text(
        json.dumps(result.model_dump(mode="json"), indent=4) + "\n"
    )
    np.savez(
        run_directory / ARRAYS_FILE,
        **{name: np.asarray(value) for name, value in arrays.items()},
    )
    return run_directory


def load_record(
    directory: pathlib.Path, inference_id: str
) -> inference_types.PlasmaStateInferenceResult:
    """Read one stored run's record, without its arrays.

    Every number a table reports is on the record, so a study of many runs reads them
    this way and pays for the npz only where a figure needs the posterior samples.

    Args:
        directory: Root of the store.
        inference_id: Id of the run to read.

    Returns:
        The validated record.
    """
    run_directory = directory / inference_id
    if not run_directory.is_dir():
        available = ", ".join(sorted(stored_inference_ids(directory))) or "none"
        raise FileNotFoundError(
            f"No inference {inference_id!r} in {directory}. Stored: {available}."
        )
    return inference_types.PlasmaStateInferenceResult.model_validate_json(
        (run_directory / RECORD_FILE).read_text()
    )


def load_inference(
    directory: pathlib.Path, inference_id: str
) -> tuple[inference_types.PlasmaStateInferenceResult, dict[str, np.ndarray]]:
    """Read one stored run back.

    Args:
        directory: Root of the store.
        inference_id: Id of the run to read.

    Returns:
        The validated record and its arrays.
    """
    result = load_record(directory, inference_id)
    with np.load(directory / inference_id / ARRAYS_FILE) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return result, arrays


def save_placement(
    directory: pathlib.Path,
    placement: diagnostic_design_point.DiagnosticPlacement,
) -> str:
    """Store a placement under its content hash, once; an equal one is already there.

    Args:
        directory: Root of the placement store.
        placement: The instrument to store.

    Returns:
        The placement id, which a record carries as `diagnostics_placement_id`.
    """
    placement_id = placement.placement_id
    path = directory / placement_id / PLACEMENT_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(placement.model_dump(mode="json"), indent=4) + "\n"
        )
    return placement_id


def load_placement(
    directory: pathlib.Path, placement_id: str
) -> diagnostic_design_point.DiagnosticPlacement:
    """Read one stored placement back.

    Args:
        directory: Root of the placement store.
        placement_id: Id of the placement to read.

    Returns:
        The validated placement.
    """
    path = directory / placement_id / PLACEMENT_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No diagnostic placement {placement_id!r} in {directory}."
        )
    return diagnostic_design_point.DiagnosticPlacement.model_validate_json(
        path.read_text()
    )


def stored_inference_ids(directory: pathlib.Path) -> list[str]:
    """Ids of every run in a store.

    Args:
        directory: Root of the store.

    Returns:
        The ids, sorted; empty when the store does not exist yet.
    """
    if not directory.is_dir():
        return []
    return sorted(
        path.name for path in directory.iterdir() if (path / RECORD_FILE).is_file()
    )


def summarize_store(directory: pathlib.Path) -> list[dict]:
    """One summary row per stored run, so a notebook can show what is available.

    Args:
        directory: Root of the store.

    Returns:
        A row per run, newest first, with the fields that identify it.
    """
    rows = []
    for inference_id in stored_inference_ids(directory):
        record = json.loads((directory / inference_id / RECORD_FILE).read_text())
        rows.append(
            {
                "inference_id": inference_id,
                "study_name": record.get("study_name", ""),
                "study_label": record.get("study_label", ""),
                "generated": record.get("generated", ""),
                "method": record["settings"]["method"],
                "profile_family": record["settings"]["profile_family"],
                "truth_name": record.get("truth_name", ""),
                "surrogate_name": record.get("surrogate_name", ""),
                "total_time_seconds": record.get("total_time_seconds", 0.0),
            }
        )
    return sorted(rows, key=lambda row: row["generated"], reverse=True)
