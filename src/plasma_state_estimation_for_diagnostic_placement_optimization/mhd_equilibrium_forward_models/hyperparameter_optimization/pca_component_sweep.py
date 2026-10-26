r"""Train one surrogate per PCA width, to choose how many components the target keeps.

Each width is scored against the round trip of its own representation, so the table says
whether the representation or the network is the limit. Writes `pca_component_sweep.csv`
into the results area, which `results.ipynb` then reads without retraining.

    S=src/plasma_state_estimation_for_diagnostic_placement_optimization
    S=$S/mhd_equilibrium_forward_models/hyperparameter_optimization
    python $S/pca_component_sweep.py --n-workers 4
"""

import os

# Set before numpy and JAX are imported, here and in every spawned worker, which
# re-imports this module to reach `score_pca_width`: a threaded BLAS under each of
# several concurrent trainings would only oversubscribe the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import concurrent.futures
import functools
import logging
import multiprocessing
import pathlib
import time
from typing import NamedTuple

import numpy as np
import pandas as pd
import sklearn.decomposition
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_types,
    dataset_utils,
    evaluation_utils,
    mlp_types,
    mlp_utils,
    preprocessing_utils,
    surrogate_utils,
    training_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models.hyperparameter_optimization import (  # noqa: E501
    hyperparameter_optimization_utils,
)

# `score_pca_width` is module level and takes the dataset by path for the same two
# reasons `hyperparameter_optimization_utils.train_node` does: stdlib pickle
# cannot send a closure, and the transformed arrays are tens of megabytes.

logger = logging.getLogger(__name__)

SWEEP_FILE = "pca_component_sweep.csv"
"""Name of the table the sweep writes, so the notebook can plot without retraining."""

NO_PCA_WIDTH = 0
"""Sentinel width meaning "no PCA at all", stored as the full target width in the table.

PCA cannot return more components than `min(n_training_rows, n_y)`, so the full target
width is only reachable by skipping the PCA entirely.
"""

SWEEP_WIDTHS = (
    200,
    400,
    600,
    800,
    958,
    1116,
    1244,
    1431,
    1588,
    1746,
    1903,
    2061,
    2218,
    2373,
    NO_PCA_WIDTH,
)
"""Widths the sweep covers: a spread up to the training rank, plus the no-PCA end."""

N_EPOCHS = 400
"""Epochs each width trains for, enough for the data-only stage to plateau."""


class SweepRow(NamedTuple):
    """One PCA width: what the network reached, and the floor it worked against."""

    requested_components: int
    """What was asked for; `NO_PCA_WIDTH` means the PCA stage was skipped."""

    latent_width: int
    """What the network actually predicted, after any clamping to the data rank."""

    validation_mse: float
    """Best validation total in this width's own latent space.

    Not comparable across rows: every row measures it in a different space, and a
    narrower space is an easier one.
    """

    n_epochs: int
    n_parameters: int
    train_seconds: float

    flux_surface_error_m: float
    """Mean flux-surface error of the trained network, in metres."""

    flux_surface_error_floor_m: float
    """The same metric for the exact solution round-tripped through this PCA."""

    iota_rmse: float
    iota_rmse_floor: float
    fnorm: float
    """Mean normalized force residual of the trained network."""

    fnorm_floor: float
    """The same for the round-tripped exact solution: the best this width allows."""

    fnorm_target: float
    """The DESC solves' own residual, identical in every row, so a consistency check."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=True,
        help="Widths trained concurrently. Each worker holds its own copy of the "
        "dataset, so 4 is comfortable on a 62 GB machine.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_sweep(
        widths=list(SWEEP_WIDTHS),
        data_directory=hyperparameter_optimization_utils.DATA_DIRECTORY,
        output_directory=hyperparameter_optimization_utils.RESULTS_DIRECTORY,
        n_epochs=N_EPOCHS,
        n_workers=arguments.n_workers,
    )


def run_sweep(
    widths: list[int],
    data_directory: pathlib.Path,
    output_directory: pathlib.Path,
    dataset_name: str = dataset_utils.DEFAULT_DATASET,
    architecture: mlp_types.MLPArchitecture | None = None,
    n_epochs: int = N_EPOCHS,
    n_train_rows: int | None = None,
    n_evaluation_samples: int = 10,
    seed: int = 0,
    n_workers: int = 4,
) -> pd.DataFrame:
    """Train and score one surrogate per PCA width, in parallel.

    One width per worker process. Each trains a fresh network and then scores both
    it and the round-tripped exact solution, so every row carries model and floor.

    Args:
        widths: Latent widths to try; `NO_PCA_WIDTH` skips the PCA stage.
        data_directory: Directory holding the dataset parquet and its settings JSON.
        output_directory: Directory the table is written to.
        dataset_name: Stem of both of those files.
        architecture: Network to hold fixed across widths; a default if None.
        n_epochs: Epochs each network trains for.
        n_train_rows: Train on this many rows only, which also caps the reachable width.
        n_evaluation_samples: Validation equilibria each row is scored on.
        seed: Seed shared by every width, so they differ only in the width.
        n_workers: Widths trained concurrently.

    Returns:
        One row per width, in the order given.
    """
    if architecture is None:
        architecture = mlp_types.MLPArchitecture(
            hidden_layer_widths=[512, 512, 512], activation="leaky_relu"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / SWEEP_FILE

    worker = functools.partial(
        score_pca_width,
        data_directory=data_directory,
        dataset_name=dataset_name,
        architecture=architecture,
        n_epochs=n_epochs,
        n_train_rows=n_train_rows,
        n_evaluation_samples=n_evaluation_samples,
        seed=seed,
    )
    logger.info(
        "Sweeping %d PCA widths over %d workers, %d epochs each.",
        len(widths),
        n_workers,
        n_epochs,
    )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=context
    ) as pool:
        rows = list(pool.map(worker, widths))

    frame = pd.DataFrame(rows, columns=list(SweepRow._fields))
    frame.to_csv(path, index=False)
    logger.info("Wrote %s.", path)
    return frame


def explained_variance_curve(
    data_directory: pathlib.Path,
    dataset_name: str = dataset_utils.DEFAULT_DATASET,
) -> pd.DataFrame:
    """Cumulative explained variance of the target PCA against the number of components.

    Fitted on the training rows through the same normalization training uses, so a
    component count read off this curve is the count `n_pca_components` would select.

    The sum is accumulated in float64 deliberately. The ratios are float32, and summing
    2373 of them in float32 drifts by enough to move where the curve crosses 1 - 1e-5 by
    three components.

    Args:
        data_directory: Directory holding the dataset parquet and its settings JSON.
        dataset_name: Stem of both of those files.

    Returns:
        One row per component count, from 1 to the rank of the training split.
    """
    _, dataset, split = dataset_utils.load_dataset_with_split(
        data_directory, dataset_name
    )
    train_rows = np.asarray(split.train)
    pressure = np.asarray(dataset.pressure)[train_rows]
    y = np.asarray(dataset.y)[train_rows]
    preprocessor = preprocessing_utils.fit_preprocessor(pressure, y, dataset.basis)
    pca = sklearn.decomposition.PCA(n_components=None)
    pca.fit(np.asarray(preprocessing_utils.transform_y(preprocessor, y)))
    cumulative = np.cumsum(pca.explained_variance_ratio_, dtype=np.float64)
    return pd.DataFrame(
        {
            "n_components": np.arange(1, cumulative.size + 1),
            "explained_variance": cumulative,
        }
    )


def score_pca_width(
    requested_components: int,
    data_directory: pathlib.Path,
    dataset_name: str,
    architecture: mlp_types.MLPArchitecture,
    n_epochs: int,
    n_train_rows: int | None,
    n_evaluation_samples: int,
    seed: int,
) -> SweepRow:
    """Train and score one surrogate at one PCA width, in this process.

    Args:
        requested_components: Latent width; `NO_PCA_WIDTH` skips the PCA stage.
        data_directory: Directory holding the dataset parquet and its settings JSON.
        dataset_name: Stem of both of those files.
        architecture: Network shape, held fixed across widths.
        n_epochs: Epochs to train for.
        n_train_rows: Train on this many rows only.
        n_evaluation_samples: Validation equilibria to score on.
        seed: Seed of the initialization and the batch order.

    Returns:
        The row for this width.
    """
    _, dataset, split = dataset_utils.load_dataset_with_split(
        data_directory, dataset_name, n_train_rows=n_train_rows
    )

    n_pca_components = (
        None if requested_components == NO_PCA_WIDTH else int(requested_components)
    )
    settings = mlp_types.MLPSettings(
        architecture=architecture,
        n_pca_components=n_pca_components,
        seed=seed,
        log_every_n_epochs=10**6,
    )
    curriculum = mlp_types.CurriculumLearningConfig(
        stages=[
            mlp_types.CurriculumStepConfig(
                loss_config=mlp_types.FinetuneLossConfig(use_data=True),
                n_epochs=n_epochs,
                patience=max(1, n_epochs // 5),
                min_epochs=n_epochs // 2,
                learning_rate=1e-3,
                label="s1_data",
            )
        ]
    )
    started = time.monotonic()
    result = training_utils.train_mlp_curriculum(curriculum, settings, dataset, split)
    train_seconds = time.monotonic() - started

    scored = dataset_utils.slice_dataset(
        dataset, np.asarray(split.validation)[:n_evaluation_samples]
    )
    targets = dataset_utils.target_equilibria(scored)
    grids = evaluation_utils.build_evaluation_grids(scored.basis)
    pressure = np.asarray(scored.pressure)
    mlp = mlp_utils.build_mlp(architecture, n_outputs=result.preprocessor.n_latent)

    predicted = np.asarray(
        mlp_utils.predict_y(mlp, result.params, result.preprocessor, pressure)
    )
    round_tripped = np.asarray(
        preprocessing_utils.inverse_transform_y(
            result.preprocessor,
            preprocessing_utils.transform_y(
                result.preprocessor, np.asarray(scored.y, dtype=np.float32)
            ),
        )
    )
    model = _mean_metrics(predicted, targets, scored, grids)
    floor = _mean_metrics(round_tripped, targets, scored, grids)

    logger.info(
        "width %s: fnorm %.3e (floor %.3e), flux %.3e m (floor %.3e)",
        result.preprocessor.n_latent,
        model["fnorm"],
        floor["fnorm"],
        model["flux"],
        floor["flux"],
    )
    return SweepRow(
        requested_components=int(requested_components),
        latent_width=int(result.preprocessor.n_latent),
        validation_mse=float(result.validation_total),
        n_epochs=int(result.n_epochs),
        n_parameters=mlp_utils.count_parameters(result.params),
        train_seconds=float(train_seconds),
        flux_surface_error_m=model["flux"],
        flux_surface_error_floor_m=floor["flux"],
        iota_rmse=model["iota"],
        iota_rmse_floor=floor["iota"],
        fnorm=model["fnorm"],
        fnorm_floor=floor["fnorm"],
        fnorm_target=model["fnorm_target"],
    )


def _mean_metrics(
    flat_targets: np.ndarray,
    targets: list,
    scored: dataset_types.RawDataset,
    grids: evaluation_utils.EvaluationGrids,
) -> dict[str, float]:
    """Score a batch of flat target vectors and average each metric.

    Args:
        flat_targets: One predicted or reconstructed flat vector per scored sample.
        targets: The DESC equilibria they should reproduce.
        scored: The samples, supplying the basis.
        grids: Evaluation grids for that basis.

    Returns:
        The mean of each metric this sweep reports.
    """
    samples = [
        evaluation_utils.evaluate_sample(
            surrogate_utils.unpack_flat_target(scored.basis, flat_targets[row]),
            targets[row],
            scored.basis,
            grids,
        )
        for row in range(scored.n_samples)
    ]
    return {
        "flux": float(np.mean([s.flux_surface_error_m for s in samples])),
        "iota": float(np.mean([s.iota_computed_rmse for s in samples])),
        "fnorm": float(np.mean([s.fnorm_predicted for s in samples])),
        "fnorm_target": float(np.mean([s.fnorm_target for s in samples])),
    }


if __name__ == "__main__":
    main()
