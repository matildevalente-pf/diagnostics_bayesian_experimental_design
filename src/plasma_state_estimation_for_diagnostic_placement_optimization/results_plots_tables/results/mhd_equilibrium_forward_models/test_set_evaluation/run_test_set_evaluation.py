r"""Test set evaluation of trained models.

Every argument has a default, so the command is usually just the path to this script.

    python src/plasma_state_estimation_for_diagnostic_placement_optimization/results_plots_tables/results/mhd_equilibrium_forward_models/test_set_evaluation/run_test_set_evaluation.py \\
        --models equilibrium_interpolator 20260908_141134_mlp_data_pca_hand_tuned \\
            20260907_081512_mlp_data_fnorm_only_pca_hand_tuned \\
        --dataset equilibrium_a_3k_spline_input \\
        --data-dir data \\
        --models-dir src/plasma_state_estimation_for_diagnostic_placement_optimization/mhd_equilibrium_forward_models/trained_models \\
        --output-dir src/plasma_state_estimation_for_diagnostic_placement_optimization/results_plots_tables/results/mhd_equilibrium_forward_models/test_set_evaluation/tables \\
        --n-samples 296 \\
        --n-workers 20
"""

import os

# Set before numpy and JAX are imported, here and in every spawned worker, which re-imports
# this module: the samples are shared out over as many processes as there are cores, so a
# threaded BLAS underneath each would only oversubscribe the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import math
import multiprocessing
import pathlib
import queue
import time

import jax
import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E402, E501
    dataset_utils,
    evaluation_utils,
    model_storage_utils,
    surrogate_utils,
)

logger = logging.getLogger(__name__)

METRICS_FILE = "mhd_surrogate_test_set_metrics.csv"
"""One row per model, the table the comparison figures are built from."""

PER_SAMPLE_FILE = "mhd_surrogate_test_set_per_sample.csv"
"""One row per model and test sample, for the distributions and the worst case."""

MIN_SHARD_SIZE = 8
"""Fewest samples worth giving a process, which spends ~45 s importing and warming DESC."""

SAMPLE_COLUMNS = (
    "flux_surface_error_m",
    "magnetic_axis_error_m",
    "iota_computed_rmse",
    "iota_predicted_rmse",
    "field_strength_relative_rmse",
    "psi_relative_error",
    "fnorm_predicted",
    "fnorm_target",
)
"""The per-sample metrics, in the order both tables list them."""

_MODULE_DIRECTORY = pathlib.Path(__file__).parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Stems of the stored models to score. Omit to be shown what is on disk, "
        "with each model's description, and asked which to take.",
    )
    parser.add_argument(
        "--dataset",
        default=dataset_utils.DEFAULT_DATASET,
        help="Stem of the dataset parquet and its settings file.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_PACKAGE_DIRECTORY.parents[1] / "data",
        help="Directory holding the dataset parquet and its settings JSON.",
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=_PACKAGE_DIRECTORY
        / "mhd_equilibrium_forward_models"
        / "trained_models",
        help="Directory holding the stored models.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_MODULE_DIRECTORY / "tables",
        help="Directory the two CSVs are written to.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Score this many test samples only, to check the plumbing.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="Processes to score in; the samples are shared out between them.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    settings_path = arguments.data_dir / f"{arguments.dataset}_settings.json"
    available = sorted(path.stem for path in arguments.models_dir.glob("*.json"))
    if not available:
        raise SystemExit(f"No stored models in {arguments.models_dir}.")
    models = arguments.models or prompt_for_models(arguments.models_dir, available)
    unknown = [name for name in models if name not in available]
    if unknown:
        raise SystemExit(f"No such stored model: {', '.join(unknown)}.")
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    provenance = {
        "generated": generated,
        "dataset": f"{arguments.dataset}.parquet",
        "settings_digest": hashlib.sha256(settings_path.read_bytes()).hexdigest()[:16],
    }

    metadata = dataset_utils.load_metadata(settings_path)
    dataset = dataset_utils.load_dataset(
        arguments.data_dir / f"{arguments.dataset}.parquet", metadata
    )
    test_rows = np.asarray(dataset_utils.split_indices(dataset, metadata).test)
    if arguments.n_samples is not None:
        test_rows = test_rows[: arguments.n_samples]
    del dataset
    shards = [
        (model, rows)
        for model in models
        for rows in _shard_rows(test_rows, len(models), arguments.n_workers)
    ]
    logger.info(
        "Scoring %s on %d samples of %s, in %d shards over %d workers.",
        ", ".join(models),
        len(test_rows),
        provenance["dataset"],
        len(shards),
        arguments.n_workers,
    )

    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        scored_samples = manager.Queue()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=arguments.n_workers, mp_context=context
        ) as pool:
            futures = [
                pool.submit(
                    evaluate_shard,
                    model,
                    rows,
                    arguments.models_dir,
                    arguments.data_dir,
                    arguments.dataset,
                    scored_samples,
                )
                for model, rows in shards
            ]
            _report_progress(
                futures, scored_samples, sum(len(rows) for _, rows in shards)
            )
            scored = [future.result() for future in futures]
    results = [
        _merge_shards([shard for (model, _), shard in zip(shards, scored) if model == name])
        for name in models
    ]

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = arguments.output_dir / METRICS_FILE
    per_sample_path = arguments.output_dir / PER_SAMPLE_FILE
    _write_rows(
        metrics_path,
        [
            {**provenance, **_summarize(model, result)}
            for model, result in zip(models, results, strict=True)
        ],
    )
    _write_rows(
        per_sample_path,
        [
            {
                **provenance,
                "model": model,
                "equilibrium_id": equilibrium_id,
                **dict(zip(SAMPLE_COLUMNS, sample, strict=True)),
            }
            for model, result in zip(models, results, strict=True)
            for equilibrium_id, sample in zip(
                result["equilibrium_ids"], result["samples"], strict=True
            )
        ],
    )
    logger.info("Wrote %s and %s.", metrics_path, per_sample_path)


def prompt_for_models(
    models_directory: pathlib.Path, available: list[str]
) -> list[str]:
    """Show what is stored, with each model's description, and ask which to score.

    Several runs of one configuration sit in the directory side by side, told apart
    only by their timestamp and their description, so scoring the wrong one is easy and
    silent.
    Asking costs a line of typing and makes a campaign's contents a deliberate choice.

    Args:
        models_directory: Directory holding the stored models.
        available: Stems found there, in order.

    Returns:
        The stems to score, in the order the answer named them.
    """
    print(f"Stored models in {models_directory}:")
    for index, name in enumerate(available):
        stored = json.loads((models_directory / f"{name}.json").read_text())
        description = stored.get("description") or stored.get("notes", "")
        print(f"  {index:2d}  {name}")
        print(f"      {description}" if description else "      (no description)")
    answer = input("Score which? Indices or stems, space separated, blank for all: ")
    tokens = answer.replace(",", " ").split()
    if not tokens:
        return available
    chosen = []
    for token in tokens:
        if not token.isdigit():
            chosen.append(token)
            continue
        index = int(token)
        if index >= len(available):
            raise SystemExit(
                f"Index {index} is out of range; the models are numbered 0 to "
                f"{len(available) - 1}."
            )
        chosen.append(available[index])
    return chosen


def evaluate_shard(
    model: str,
    rows: np.ndarray,
    models_dir: pathlib.Path,
    data_dir: pathlib.Path,
    dataset_name: str,
    scored_samples: queue.Queue | None = None,
) -> dict:
    """Score one model on part of the test split, in its own process.

    Args:
        model: Name of the stored model.
        rows: Dataset row indices this shard covers.
        models_dir: Directory holding the model.
        data_dir: Directory holding the dataset.
        dataset_name: Stem of the dataset parquet and settings file.
        scored_samples: Queue to announce each finished sample on, so the parent can report
            the progress of the whole run; None to score without reporting.

    Returns:
        The per-sample metrics of this shard, the ids they belong to, the forward times and
        the model's size.
    """
    metadata = dataset_utils.load_metadata(
        data_dir / f"{dataset_name}_settings.json"
    )
    dataset = dataset_utils.load_dataset(
        data_dir / f"{dataset_name}.parquet", metadata
    )
    test_set = dataset_utils.slice_dataset(dataset, rows)
    targets = dataset_utils.target_equilibria(test_set)
    grids = evaluation_utils.build_evaluation_grids(test_set.basis)
    pressure = np.asarray(test_set.pressure)

    surrogate = model_storage_utils.load_surrogate(models_dir, model, data_dir)
    dataset_utils.assert_compatible_bases(surrogate.basis, metadata.basis)
    # Jitted, because that is how the inference loop this surrogate exists for calls it, and
    # the latency in the comparison table has to be the latency that loop would see.
    predict = jax.jit(surrogate.predict)
    samples = []
    forward_seconds = []
    for row in range(test_set.n_samples):
        started = time.monotonic()
        y = np.asarray(predict(pressure[row]))
        forward_seconds.append(time.monotonic() - started)
        samples.append(
            evaluation_utils.evaluate_sample(
                surrogate_utils.unpack_flat_target(test_set.basis, y),
                targets[row],
                test_set.basis,
                grids,
            )
        )
        if scored_samples is not None:
            scored_samples.put(1)
    logger.info("Scored %s on %d samples.", model, len(samples))
    return {
        "equilibrium_ids": test_set.equilibrium_ids,
        "samples": samples,
        "forward_seconds": forward_seconds,
        "n_parameters": surrogate.n_parameters,
        "kind": surrogate.kind,
    }


def _report_progress(
    futures: list[concurrent.futures.Future],
    scored_samples: queue.Queue,
    total: int,
) -> None:
    """Log how far the run has got, until every shard is finished.

    The workers are silent until they finish, and there are as many of them as there are
    cores, so without this a run of minutes prints nothing at all and then everything at once.
    Each worker announces every sample it scores, and this counts them.

    Args:
        futures: The shards, so the loop can stop if one of them dies.
        scored_samples: Queue the workers announce on.
        total: Samples the whole run will score.
    """
    started = time.monotonic()
    scored = 0
    # A progress line every five percent of the run.
    next_report = 5.0
    while scored < total:
        try:
            scored_samples.get(timeout=1.0)
        except queue.Empty:
            if all(future.done() for future in futures):
                return
            continue
        scored += 1
        percent = 100.0 * scored / total
        if percent < next_report and scored < total:
            continue
        next_report = percent + 5.0
        elapsed = time.monotonic() - started
        logger.info(
            "%d of %d samples, %.0f%%, %.0f s elapsed, about %.0f s left.",
            scored,
            total,
            percent,
            elapsed,
            elapsed * (total - scored) / scored,
        )


def _shard_rows(
    test_rows: np.ndarray, n_models: int, n_workers: int
) -> list[np.ndarray]:
    """Split the test rows into contiguous shards, one per worker per model.

    A shard pays for its own process: importing DESC and compiling the first `compute` costs
    the better part of a minute, which is why shards are not allowed to fall below
    `MIN_SHARD_SIZE` samples however many workers are free.

    Args:
        test_rows: Row indices of the whole test split.
        n_models: Number of models being scored, which share the workers.
        n_workers: Processes available.

    Returns:
        The shards, together covering `test_rows` in order.
    """
    per_model = max(1, math.ceil(n_workers / n_models))
    affordable = max(1, math.ceil(len(test_rows) / MIN_SHARD_SIZE))
    n_shards = min(per_model, affordable)
    return [shard for shard in np.array_split(test_rows, n_shards) if len(shard)]


def _merge_shards(shards: list[dict]) -> dict:
    """Reassemble one model's shards into the result of a whole test set.

    Args:
        shards: That model's shards, in test-set order.

    Returns:
        The same shape of result a single unsharded run would have produced.
    """
    return {
        "equilibrium_ids": [
            equilibrium_id
            for shard in shards
            for equilibrium_id in shard["equilibrium_ids"]
        ],
        "samples": [sample for shard in shards for sample in shard["samples"]],
        "forward_seconds": [
            seconds for shard in shards for seconds in shard["forward_seconds"]
        ],
        "n_parameters": shards[0]["n_parameters"],
        "kind": shards[0]["kind"],
    }


def _summarize(model: str, result: dict) -> dict:
    """Reduce one model's per-sample metrics to the row of the comparison table.

    The force residual is summarized by its median and its ninetieth percentile as well as
    its mean, because its distribution has a heavy tail and the mean sits inside it.

    Args:
        model: Name of the model.
        result: What the model's shards produced, merged.

    Returns:
        One row of the comparison table.
    """
    columns = {
        name: np.array([getattr(sample, name) for sample in result["samples"]])
        for name in SAMPLE_COLUMNS
    }
    row = {
        "model": model,
        "kind": result["kind"],
        "n_samples": len(result["samples"]),
        "n_parameters": result["n_parameters"],
        # The median, not the mean: the first call in each shard pays for tracing and
        # compiling the forward pass, which is not what the inference loop this model is for
        # will see.
        "inference_ms": 1e3 * float(np.median(result["forward_seconds"])),
    }
    row.update({name: float(np.mean(values)) for name, values in columns.items()})
    for name in ("fnorm_predicted", "fnorm_target"):
        row[f"{name}_p50"] = float(np.percentile(columns[name], 50))
        row[f"{name}_p90"] = float(np.percentile(columns[name], 90))
    row["flux_surface_error_m_worst"] = float(np.max(columns["flux_surface_error_m"]))
    return row


def _write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    """Write a list of identically keyed rows out as a CSV.

    Args:
        path: File to write.
        rows: The rows, at least one.
    """
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
