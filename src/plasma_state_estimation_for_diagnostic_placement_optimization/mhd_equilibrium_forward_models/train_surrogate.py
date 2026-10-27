r"""Train one MLP equilibrium surrogate and store it under `trained_models/`.

    python src/plasma_state_estimation_for_diagnostic_placement_optimization/mhd_equilibrium_forward_models/train_surrogate.py \\
        --model mlp_data_fnorm_only_pca_hand_tuned --description "thesis model"
"""

import argparse
import json
import logging
import pathlib
import time

from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_types,
    dataset_utils,
    mlp_types,
    model_configs,
    model_storage_utils,
    training_utils,
)

logger = logging.getLogger(__name__)

_MODULE_DIRECTORY = pathlib.Path(__file__).parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parent

MODELS_DIRECTORY = _MODULE_DIRECTORY / "trained_models"
"""Directory trained models are written to."""

TRAINING_RUNS_DIRECTORY = (
    _PACKAGE_DIRECTORY
    / "results_plots_tables"
    / "results"
    / "mhd_equilibrium_forward_models"
    / "models_training"
)
"""Directory each training run's history and summary are written under."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(model_configs.MODEL_CONFIGURATIONS),
        help="Which of the named configurations to train.",
    )
    parser.add_argument(
        "--description",
        required=True,
        help="What this run is for, in your own words. Recorded in the stored model "
        "and shown when choosing between models, which is the only thing telling two "
        "runs of one configuration apart.",
    )
    parser.add_argument(
        "--dataset",
        default=dataset_utils.DEFAULT_DATASET,
        help="Stem of the dataset parquet and its settings file in the data directory.",
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
        default=MODELS_DIRECTORY,
        help="Directory the trained model is written to.",
    )
    parser.add_argument(
        "--runs-dir",
        type=pathlib.Path,
        default=TRAINING_RUNS_DIRECTORY,
        help="Directory the per-run history and summary are written under.",
    )
    parser.add_argument(
        "--max-epochs-per-stage",
        type=int,
        default=None,
        help="Cap every stage's epoch count, for a smoke run.",
    )
    parser.add_argument(
        "--n-train-rows",
        type=int,
        default=None,
        help="Train on this many rows of the training split only, for a smoke run.",
    )
    parser.add_argument(
        "--max-fnorm-vol",
        type=float,
        default=None,
        help="Force-residual filter; defaults to the dataset's own recommendation. "
        "Changing it invalidates the pinned split.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    configuration = model_configs.MODEL_CONFIGURATIONS[arguments.model]
    settings = configuration.settings
    curriculum = training_utils.cap_curriculum_epochs(
        configuration.curriculum, arguments.max_epochs_per_stage
    )

    metadata, dataset, split = dataset_utils.load_dataset_with_split(
        arguments.data_dir,
        arguments.dataset,
        max_fnorm_vol=arguments.max_fnorm_vol,
        n_train_rows=arguments.n_train_rows,
    )
    logger.info(
        "Dataset %s: %d rows kept, split %d / %d / %d.",
        metadata.name,
        dataset.n_samples,
        len(split.train),
        len(split.validation),
        len(split.test),
    )

    result = training_utils.train_mlp_curriculum(curriculum, settings, dataset, split)
    logger.info(
        "Trained %s in %.1f s over %d epochs: train %.4e, validation total %.4e.",
        arguments.model,
        result.train_seconds,
        result.n_epochs,
        result.train_mse,
        result.validation_total,
    )

    weights_path, metadata_path, run_directory = store_training_run(
        model_name=arguments.model,
        description=arguments.description,
        configuration_description=configuration.description,
        settings=settings,
        curriculum=curriculum,
        result=result,
        dataset=dataset,
        dataset_name=metadata.name,
        max_fnorm_vol=arguments.max_fnorm_vol,
        arguments={
            key: str(value) for key, value in vars(arguments).items() if key != "model"
        },
        models_directory=arguments.models_dir,
        runs_directory=arguments.runs_dir,
    )
    logger.info("Wrote %s, %s and %s.", weights_path, metadata_path, run_directory)


def store_training_run(
    model_name: str,
    description: str,
    configuration_description: str,
    settings: mlp_types.MLPSettings,
    curriculum: mlp_types.CurriculumLearningConfig,
    result: training_utils.TrainingResult,
    dataset: dataset_types.RawDataset,
    dataset_name: str,
    max_fnorm_vol: float | None,
    arguments: dict[str, str],
    models_directory: pathlib.Path,
    runs_directory: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Write a trained model and its run record, both named by the time it finished.

    Args:
        model_name: Name the model is stored under, after the timestamp.
        description: What the run is for, recorded in the model and the run record.
        configuration_description: One line on the configuration that was trained.
        settings: Architecture, optimizer and target-space settings it trained with.
        curriculum: The stages it was trained through.
        result: What the training produced.
        dataset: The rows it was trained on, supplying the basis.
        dataset_name: Name of that dataset.
        max_fnorm_vol: Force-residual filter it was read with, None for the default.
        arguments: Any further settings of the run, recorded as strings.
        models_directory: Directory the model is written to.
        runs_directory: Directory the run record is written under.

    Returns:
        The weights file, the metadata file and the run directory.
    """
    generated = time.strftime("%Y-%m-%d")
    # One timestamp for both, so a model file and its run directory share a name and
    # no run ever overwrites the one before it.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{model_name}"
    run_directory = runs_directory / stem
    run_directory.mkdir(parents=True, exist_ok=True)
    training_utils.write_history_csv(run_directory / "history.csv", result.history)
    (run_directory / "config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "stem": stem,
                "description": description,
                "configuration_description": configuration_description,
                "dataset": dataset_name,
                "max_fnorm_vol": max_fnorm_vol,
                "settings": settings.model_dump(),
                "curriculum": curriculum.model_dump(),
                "arguments": arguments,
            },
            indent=4,
        )
        + "\n"
    )
    (run_directory / "summary.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "stem": stem,
                "description": description,
                "generated": generated,
                "n_epochs": result.n_epochs,
                "train_mse": result.train_mse,
                "validation_total": result.validation_total,
                "train_seconds": result.train_seconds,
                "stages": [
                    {
                        "label": stage.label,
                        "n_epochs": produced.n_epochs,
                        "validation_total": produced.selected.validation_total,
                        "validation_data": produced.selected.validation_data,
                        "validation_fnorm": produced.selected.validation_fnorm,
                        "selection_term": produced.selection_term,
                        "fnorm_weight": produced.fnorm_weight,
                    }
                    for stage, produced in zip(
                        curriculum.stages, result.stages, strict=True
                    )
                ],
            },
            indent=4,
        )
        + "\n"
    )

    weights_path, metadata_path = model_storage_utils.save_mlp_model(
        models_directory,
        stem,
        model_name,
        description,
        settings,
        curriculum,
        dataset.basis,
        result,
        dataset_name=dataset_name,
        # Relative to the package, so the record stays valid in any checkout.
        run_directory=str(
            run_directory.relative_to(_PACKAGE_DIRECTORY)
            if run_directory.is_relative_to(_PACKAGE_DIRECTORY)
            else run_directory
        ),
        generated=generated,
    )
    return weights_path, metadata_path, run_directory


if __name__ == "__main__":
    main()
