"""Grid search over the MLP surrogate's hyperparameters, then a force-stage comparison.

S=src/plasma_state_estimation_for_diagnostic_placement_optimization
S=$S/mhd_equilibrium_forward_models/hyperparameter_optimization
python $S/hyperparameter_optimization_utils.py
python $S/hyperparameter_optimization_utils.py --force-stage
"""

import os

# Before numpy and JAX are imported, here and in every spawned worker: a threaded BLAS
# under each of several concurrent trainings would only oversubscribe the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import concurrent.futures
import functools
import itertools
import logging
import multiprocessing
import pathlib
import time
from typing import Any, NamedTuple

import pandas as pd
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_utils,
    mlp_types,
    mlp_utils,
    model_configs,
    train_surrogate,
    training_utils,
)

logger = logging.getLogger(__name__)


class GridNode(NamedTuple):
    """One combination of the searched hyperparameters."""

    activation: mlp_types.ActivationName
    width: int
    """Neurons in every hidden layer."""

    depth: int
    """Number of hidden layers."""

    batch_size: int


MODEL = "mlp_data_pca_hand_tuned"
"""Configuration searched: the data stage of the thesis model, itself one grid node."""

GRID: dict[str, tuple[Any, ...]] = {
    "activation": ("relu", "leaky_relu", "gelu", "tanh"),
    "width": (256, 512, 1024),
    "depth": (2, 3, 4),
    "batch_size": (32, 64, 128),
}
"""Values tried per hyperparameter, every combination trained: 108 nodes."""

N_WORKERS = 5
"""Trainings run concurrently, each in its own process with its own copy of the data."""

MAX_EPOCHS_PER_STAGE: int | None = None
"""Epoch cap per stage; None trains every node through the data stage of `MODEL`."""

FORCE_STAGE_MODEL = "mlp_data_fnorm_only_pca_hand_tuned"
"""Pipeline of the thesis model, data stage then force stage, run on the nodes below."""

FORCE_STAGE_NODES = (
    GridNode("tanh", 256, 4, 128),
    GridNode("tanh", 256, 4, 64),
    GridNode("tanh", 256, 3, 64),
    GridNode("leaky_relu", 512, 3, 32),
    GridNode("leaky_relu", 256, 4, 64),
)
"""Ranks 1, 2, 3, 71 (the thesis model) and 17 (the best leaky ReLU) of the grid run
`searches/20261004_195650_mlp_data_pca_hand_tuned`, ranked again on the force term."""

TRIALS_FILE = "trials.csv"

_PACKAGE_DIRECTORY = pathlib.Path(__file__).parents[2]

DATA_DIRECTORY = _PACKAGE_DIRECTORY.parents[1] / "data"
"""Directory holding the dataset parquet and its settings JSON."""

RESULTS_DIRECTORY = (
    _PACKAGE_DIRECTORY
    / "results_plots_tables"
    / "results"
    / "mhd_equilibrium_forward_models"
    / "hyperparameter_optimization"
)
"""Directory holding both studies' results, read by `results.ipynb`."""

RUNS_DIRECTORY = RESULTS_DIRECTORY / "searches"
"""Directory each search's own timestamped run folder is created under."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force-stage",
        action="store_true",
        help="Train FORCE_STAGE_NODES through FORCE_STAGE_MODEL and store each model, "
        "instead of running the grid.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if arguments.force_stage:
        run_grid_search(
            list(FORCE_STAGE_NODES),
            RUNS_DIRECTORY / f"{timestamp}_{FORCE_STAGE_MODEL}",
            n_workers=N_WORKERS,
            max_epochs=None,
            model_name=FORCE_STAGE_MODEL,
            models_directory=train_surrogate.MODELS_DIRECTORY,
        )
        return
    nodes = grid_nodes(GRID)
    thesis_node = node_of_configuration(model_configs.MODEL_CONFIGURATIONS[MODEL])
    if thesis_node not in nodes:
        raise ValueError(f"The grid must contain {MODEL}'s own node, {thesis_node}.")
    run_grid_search(
        nodes,
        RUNS_DIRECTORY / f"{timestamp}_{MODEL}",
        n_workers=N_WORKERS,
        max_epochs=MAX_EPOCHS_PER_STAGE,
    )


def run_grid_search(
    nodes: list[GridNode],
    output_directory: pathlib.Path,
    n_workers: int,
    max_epochs: int | None,
    model_name: str = MODEL,
    data_directory: pathlib.Path = DATA_DIRECTORY,
    dataset_name: str = dataset_utils.DEFAULT_DATASET,
    models_directory: pathlib.Path | None = None,
    training_runs_directory: pathlib.Path = train_surrogate.TRAINING_RUNS_DIRECTORY,
) -> pd.DataFrame:
    """Train one network per grid node, `n_workers` at a time, and tabulate the results.

    Args:
        nodes: Hyperparameter combinations to train.
        output_directory: Directory the table is written to, after every finished node.
        n_workers: Trainings run concurrently.
        max_epochs: Epoch cap per stage, or None for the full curriculum.
        model_name: Configuration every node overrides, a key of `MODEL_CONFIGURATIONS`.
        data_directory: Directory holding the dataset parquet and its settings JSON.
        dataset_name: Stem of both of those files.
        models_directory: Directory every trained model is stored in, or None.
        training_runs_directory: Directory their run records are written under.

    Returns:
        One row per node, in grid order.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    worker = functools.partial(
        train_node,
        model_name=model_name,
        data_directory=data_directory,
        dataset_name=dataset_name,
        max_epochs=max_epochs,
        models_directory=models_directory,
        training_runs_directory=training_runs_directory,
    )
    logger.info(
        "Training %d grid nodes of %s, %d at a time, into %s.",
        len(nodes),
        model_name,
        n_workers,
        output_directory,
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=context
    ) as pool:
        futures = [pool.submit(worker, trial, node) for trial, node in enumerate(nodes)]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            # Rewritten after every node, so an interrupted search keeps what it did.
            pd.DataFrame(rows).sort_values("trial").to_csv(
                output_directory / TRIALS_FILE, index=False
            )
            logger.info(
                "%3d of %d done  %s  validation %.4e  best %.4e  %.1f h elapsed",
                len(rows),
                len(nodes),
                nodes[rows[-1]["trial"]],
                rows[-1]["objective"],
                min(row["objective"] for row in rows),
                (time.monotonic() - started) / 3600.0,
            )
    trials = pd.DataFrame(rows).sort_values("trial", ignore_index=True)
    best = trials.loc[trials["objective"].idxmin()]
    logger.info("Best %.4e at trial %d:\n%s", best["objective"], best["trial"], best)
    return trials


def train_node(
    trial: int,
    node: GridNode,
    model_name: str,
    data_directory: pathlib.Path,
    dataset_name: str,
    max_epochs: int | None,
    models_directory: pathlib.Path | None,
    training_runs_directory: pathlib.Path,
) -> dict[str, Any]:
    """Train one grid node in this process and score it on the validation rows.

    Module level and handed the dataset by path, since a spawned worker cannot receive a
    closure and the arrays are tens of megabytes.

    Args:
        trial: Index of the node in the grid.
        node: Hyperparameters to train with.
        model_name: Configuration the node overrides.
        data_directory: Directory holding the dataset parquet and its settings JSON.
        dataset_name: Stem of both of those files.
        max_epochs: Epoch cap per stage, or None for the full curriculum.
        models_directory: Directory the trained model is stored in, or None.
        training_runs_directory: Directory its run record is written under.

    Returns:
        The node's row of the results table.
    """
    # A spawned worker starts with logging unconfigured, and runs several nodes in turn:
    # retag its lines per node so concurrent trainings stay readable in one log.
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{node.activation} {node.width}x{node.depth} "
        f"batch {node.batch_size}] %(message)s",
        force=True,
    )
    metadata, dataset, split = dataset_utils.load_dataset_with_split(
        data_directory, dataset_name
    )
    base = model_configs.MODEL_CONFIGURATIONS[model_name]
    configuration = configuration_from_node(base, node, max_epochs)
    result = training_utils.train_mlp_curriculum(
        configuration.curriculum, configuration.settings, dataset, split
    )
    stage = result.stages[-1]
    objective = min(
        {
            "total": record.validation_total,
            "data": record.validation_data,
            "fnorm": record.validation_fnorm,
        }[stage.selection_term]
        for record in stage.history
    )
    stored_model = ""
    if models_directory is not None:
        weights_path, _, _ = train_surrogate.store_training_run(
            model_name=(
                f"{model_name}_{node.activation}_{node.width}x{node.depth}"
                f"_batch{node.batch_size}"
            ),
            description=f"Grid node {tuple(node)} trained through {model_name}.",
            configuration_description=base.description,
            settings=configuration.settings,
            curriculum=configuration.curriculum,
            result=result,
            dataset=dataset,
            dataset_name=metadata.name,
            max_fnorm_vol=None,
            arguments={},
            models_directory=models_directory,
            runs_directory=training_runs_directory,
        )
        stored_model = weights_path.stem
    return {
        "trial": trial,
        **node._asdict(),
        "learning_rate": configuration.curriculum.stages[0].learning_rate,
        "weight_decay": configuration.settings.weight_decay,
        "dropout_rate": configuration.settings.architecture.dropout_rate,
        "objective": objective,
        "objective_term": stage.selection_term,
        "n_epochs": result.n_epochs,
        "n_parameters": mlp_utils.count_parameters(result.params),
        "train_seconds": result.train_seconds,
        "is_thesis_model": node == node_of_configuration(base),
        "stored_model": stored_model,
    }


def grid_nodes(grid: dict[str, tuple[Any, ...]]) -> list[GridNode]:
    """List every combination of the grid's values, the last hyperparameter fastest.

    Args:
        grid: Values per hyperparameter, keyed by the field names of `GridNode`.

    Returns:
        The nodes.
    """
    return [
        GridNode(**dict(zip(grid, values)))
        for values in itertools.product(*grid.values())
    ]


def configuration_from_node(
    base: model_configs.ModelConfiguration, node: GridNode, max_epochs: int | None
) -> model_configs.ModelConfiguration:
    """Overlay one node on a configuration, keeping everything the grid does not vary.

    Args:
        base: The configuration searched.
        node: Hyperparameters to set.
        max_epochs: Epoch cap per stage, or None for the full curriculum.

    Returns:
        The configuration to train.
    """
    architecture = base.settings.architecture.model_copy(
        update={
            "hidden_layer_widths": [node.width] * node.depth,
            "activation": node.activation,
        }
    )
    settings = base.settings.model_copy(
        update={"architecture": architecture, "batch_size": node.batch_size}
    )
    return model_configs.ModelConfiguration(
        settings=settings,
        curriculum=training_utils.cap_curriculum_epochs(base.curriculum, max_epochs),
        description=base.description,
    )


def node_of_configuration(configuration: model_configs.ModelConfiguration) -> GridNode:
    """Read the searched hyperparameters off a configuration with equal layer widths.

    Args:
        configuration: The configuration to read.

    Returns:
        Its grid node.
    """
    architecture = configuration.settings.architecture
    return GridNode(
        activation=architecture.activation,
        width=architecture.hidden_layer_widths[0],
        depth=len(architecture.hidden_layer_widths),
        batch_size=configuration.settings.batch_size,
    )


if __name__ == "__main__":
    main()
