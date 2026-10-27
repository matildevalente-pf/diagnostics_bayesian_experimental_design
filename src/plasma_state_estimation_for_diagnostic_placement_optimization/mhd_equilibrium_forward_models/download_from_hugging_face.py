import argparse
import logging
import pathlib
import shutil
from typing import NamedTuple

import huggingface_hub
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_utils,
    train_surrogate,
)

logger = logging.getLogger(__name__)

_PACKAGE_DIRECTORY = pathlib.Path(__file__).parents[1]

DATA_DIRECTORY = _PACKAGE_DIRECTORY.parents[1] / "data"
"""Directory the datasets are read from by default."""


class Repository(NamedTuple):
    """A Hugging Face repository at a fixed revision."""

    repo_id: str
    repo_type: str
    revision: str


DATASETS_REPOSITORY = Repository(
    "proxima-fusion/diagnostics-bayesian-experimental-design-datasets",
    "dataset",
    "1867b24544342ccc081b76724701cd5bbdccf4fb",
)
SURROGATES_REPOSITORY = Repository(
    "proxima-fusion/diagnostics-bayesian-experimental-design-surrogates",
    "model",
    "c8395062702ec043415ba7646ce7d70a018fe7f7",
)

DATASETS = (
    dataset_utils.DEFAULT_DATASET,
    "equilibrium_a_interpolator_28_spline_input",
)
"""The surrogates' training dataset and the equilibrium interpolator's lattice."""

WARM_START = "equilibrium_a_warm_start.json"
"""The DESC starting point every row of both datasets was solved from."""

SURROGATES = (
    "20260907_081512_mlp_data_fnorm_only_pca_hand_tuned",
    "20260908_141134_mlp_data_pca_hand_tuned",
)

TRAINING_RUN_FILES = ("history.csv", "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=DATA_DIRECTORY,
        help="Directory the dataset parquets and settings JSONs are written to.",
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=train_surrogate.MODELS_DIRECTORY,
        help="Directory the surrogate weights and JSONs are written to.",
    )
    parser.add_argument(
        "--runs-dir",
        type=pathlib.Path,
        default=train_surrogate.TRAINING_RUNS_DIRECTORY,
        help="Directory the surrogates' training histories are written under.",
    )
    arguments = parser.parse_args()
    # Only this script's own messages: the HTTP client logs every request at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)

    for dataset in DATASETS:
        for filename in (f"{dataset}.parquet", f"{dataset}_settings.json"):
            download(DATASETS_REPOSITORY, filename, arguments.data_dir / filename)
    download(DATASETS_REPOSITORY, WARM_START, arguments.data_dir / WARM_START)
    for surrogate in SURROGATES:
        for filename in (f"{surrogate}.npz", f"{surrogate}.json"):
            download(SURROGATES_REPOSITORY, filename, arguments.models_dir / filename)
        for filename in TRAINING_RUN_FILES:
            download(
                SURROGATES_REPOSITORY,
                f"models_training/{surrogate}/{filename}",
                arguments.runs_dir / surrogate / filename,
            )


def download(repository: Repository, filename: str, target: pathlib.Path) -> None:
    """Copy one file of `repository` to `target`, unless `target` already exists."""
    if target.exists():
        logger.info("Kept the existing %s", target)
        return
    cached = huggingface_hub.hf_hub_download(
        repository.repo_id,
        filename,
        repo_type=repository.repo_type,
        revision=repository.revision,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name first: an interrupted copy would otherwise leave a
    # truncated file that the next run keeps as if it were complete.
    partial = target.with_name(f"{target.name}.partial")
    shutil.copyfile(cached, partial)
    partial.rename(target)
    logger.info("Downloaded %s", target)


if __name__ == "__main__":
    main()
