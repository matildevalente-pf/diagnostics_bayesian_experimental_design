# Reading a surrogate training dataset off disk: its sidecar settings JSON, its parquet,
# and the check that two artifacts speak the same basis.
#
# The parquet already holds every quantity computed with DESC, so loading is pure local
# I/O and a training run never imports DESC.

import json
import math
import pathlib

import numpy as np
import pandas as pd
from jaxtyping import Float, Int
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_types,
    surrogate_types,
    surrogate_utils,
)

FNORM_COLUMN = "fnorm_volume_averaged_normalized_force_error"
"""Parquet column the convergence filter is applied to."""

DEFAULT_DATASET = "equilibrium_a_3k_spline_input"
"""Dataset every model in the comparison is trained and scored on: the pressure at the
99 stored spline knots as input, derived by `derive_spline_input_dataset.py`."""


def load_metadata(settings_path: pathlib.Path) -> dataset_types.DatasetMetadata:
    """Read a dataset's sidecar settings JSON.

    Args:
        settings_path: Path of the `<dataset>_settings.json` file.

    Returns:
        The metadata, with the basis assembled from the resolution, mode tables and grids.
    """
    settings = json.loads(settings_path.read_text())
    if settings["flat_target_layout"] != surrogate_types.FLAT_TARGET_LAYOUT:
        raise ValueError(
            f"{settings_path} was written for target layout "
            f"{settings['flat_target_layout']!r}, but this package predicts "
            f"{surrogate_types.FLAT_TARGET_LAYOUT!r}."
        )
    radial, poloidal, toroidal = settings["resolution_lmn"]
    return dataset_types.DatasetMetadata(
        name=settings["dataset"],
        generated=settings["generated"],
        plasma_configuration_id=settings["plasma_configuration_id"],
        recommended_max_fnorm_vol=settings.get("recommended_max_fnorm_vol"),
        n_samples=settings["n_samples"],
        basis=surrogate_types.EquilibriumBasisSpecification(
            resolution=desc_types.DescResolution(
                radial_resolution=radial,
                poloidal_resolution=poloidal,
                toroidal_resolution=toroidal,
                n_field_periods=settings["nfp"],
                is_stellarator_symmetric=settings["stellarator_symmetric"],
                spectral_indexing=settings["spectral_indexing"],
            ),
            r_modes=np.asarray(settings["r_modes_lmn"], dtype=int),
            z_modes=np.asarray(settings["z_modes_lmn"], dtype=int),
            lambda_modes=np.asarray(settings["lambda_modes_lmn"], dtype=int),
            pressure_rho_grid=np.asarray(settings["pressure_rho_grid"], dtype=float),
            iota_rho_grid=np.asarray(settings["iota_rho_grid"], dtype=float),
        ),
        train_ids=settings.get("train_ids"),
        # The export script writes the key as `val_ids`; only the field is spelled out.
        validation_ids=settings.get("val_ids"),
        test_ids=settings.get("test_ids"),
    )


def load_dataset(
    parquet_path: pathlib.Path,
    metadata: dataset_types.DatasetMetadata,
    max_fnorm_vol: float | None = None,
) -> dataset_types.RawDataset:
    """Read a dataset parquet, keeping the converged rows in their stored order.

    Args:
        parquet_path: Path of the dataset parquet.
        metadata: Metadata of that dataset, fixing the basis the rows are read into.
        max_fnorm_vol: Force-residual threshold; rows above it, and rows whose residual is
            missing, are dropped. Defaults to the recommendation in `metadata`, or to
            keeping every row when it makes none. Pass `math.inf` to keep the whole tail.

    Returns:
        The kept rows as flat arrays.
    """
    if max_fnorm_vol is None:
        max_fnorm_vol = metadata.recommended_max_fnorm_vol
    if max_fnorm_vol is None:
        max_fnorm_vol = math.inf

    frame = pd.read_parquet(parquet_path)
    if len(frame) != metadata.n_samples:
        raise ValueError(
            f"{parquet_path} has {len(frame)} rows but {metadata.name} is recorded as "
            f"{metadata.n_samples} samples."
        )
    converged = frame[FNORM_COLUMN].notna() & (frame[FNORM_COLUMN] <= max_fnorm_vol)
    frame = frame[converged]

    return dataset_types.RawDataset(
        basis=metadata.basis,
        pressure=_stack(frame, "pressure"),
        y=np.concatenate(
            [
                _stack(frame, "R_lmn"),
                _stack(frame, "Z_lmn"),
                _stack(frame, "lambda_lmn"),
                _stack(frame, "iotaf"),
                frame["Psi"].to_numpy(dtype=float)[:, None],
            ],
            axis=1,
        ),
        pressure_knots=_stack(frame, "pressure_knots"),
        pressure_params=_stack(frame, "pressure_params"),
        fnorm_vol=frame[FNORM_COLUMN].to_numpy(dtype=float),
        beta_vol=frame["beta_volume_averaged"].to_numpy(dtype=float),
        equilibrium_ids=frame["desc_equilibrium_id"].tolist(),
    )


def split_indices(
    dataset: dataset_types.RawDataset,
    metadata: dataset_types.DatasetMetadata,
) -> dataset_types.DataSplit:
    """Resolve the pinned train/validation/test id lists into row indices of `dataset`.

    The partitions must together cover the loaded rows exactly. That equality is the
    split's provenance guarantee: it was defined over the rows that clear the recommended
    force-residual filter, so it breaks the moment a dataset is loaded at another
    threshold, and silently training on the wrong rows is what it prevents.

    Args:
        dataset: Dataset the ids are resolved against.
        metadata: Metadata holding the pinned id lists.

    Returns:
        The three partitions as row indices, each in the order of its id list.
    """
    if not metadata.has_split:
        raise ValueError(f"{metadata.name} pins no train/validation/test split.")
    assert metadata.train_ids is not None
    assert metadata.validation_ids is not None
    assert metadata.test_ids is not None

    row_of_id = {
        equilibrium_id: row
        for row, equilibrium_id in enumerate(dataset.equilibrium_ids)
    }
    split_ids = (
        set(metadata.train_ids) | set(metadata.validation_ids) | set(metadata.test_ids)
    )
    if split_ids != set(row_of_id):
        raise ValueError(
            f"The split of {metadata.name} covers {len(split_ids)} equilibria but the "
            f"loaded dataset holds {len(row_of_id)}, of which "
            f"{len(set(row_of_id) - split_ids)} are outside the split. The split was "
            f"defined at max_fnorm_vol = {metadata.recommended_max_fnorm_vol}."
        )
    return dataset_types.DataSplit(
        train=np.array([row_of_id[i] for i in metadata.train_ids], dtype=int),
        validation=np.array([row_of_id[i] for i in metadata.validation_ids], dtype=int),
        test=np.array([row_of_id[i] for i in metadata.test_ids], dtype=int),
    )


def load_dataset_with_split(
    data_directory: pathlib.Path,
    dataset_name: str,
    max_fnorm_vol: float | None = None,
    n_train_rows: int | None = None,
) -> tuple[
    dataset_types.DatasetMetadata,
    dataset_types.RawDataset,
    dataset_types.DataSplit,
]:
    """Read a dataset and its pinned split from the directory holding both of its files.

    Args:
        data_directory: Directory holding the dataset parquet and its settings JSON.
        dataset_name: Stem of both of those files.
        max_fnorm_vol: Force-residual threshold; defaults to the dataset's own
            recommendation.
        n_train_rows: Keep this many rows of the train and validation splits only, for a
            smoke run; None keeps all of them.

    Returns:
        The metadata, the kept rows, and the split as row indices.
    """
    metadata = load_metadata(data_directory / f"{dataset_name}_settings.json")
    dataset = load_dataset(
        data_directory / f"{dataset_name}.parquet",
        metadata,
        max_fnorm_vol=max_fnorm_vol,
    )
    split = split_indices(dataset, metadata)
    if n_train_rows is not None:
        split = dataset_types.DataSplit(
            train=np.asarray(split.train)[:n_train_rows],
            validation=np.asarray(split.validation)[:n_train_rows],
            test=split.test,
        )
    return metadata, dataset, split


def slice_dataset(
    dataset: dataset_types.RawDataset,
    indices: Int[NpOrJaxArray, " n_selected"],
) -> dataset_types.RawDataset:
    """Take a subset of a dataset's rows, in the order `indices` gives them.

    Args:
        dataset: Dataset to take rows from.
        indices: Row indices to keep.

    Returns:
        A dataset holding those rows, in the same basis.
    """
    rows = np.asarray(indices, dtype=int)
    return dataset_types.RawDataset(
        basis=dataset.basis,
        pressure=np.asarray(dataset.pressure)[rows],
        y=np.asarray(dataset.y)[rows],
        pressure_knots=np.asarray(dataset.pressure_knots)[rows],
        pressure_params=np.asarray(dataset.pressure_params)[rows],
        fnorm_vol=np.asarray(dataset.fnorm_vol)[rows],
        beta_vol=np.asarray(dataset.beta_vol)[rows],
        equilibrium_ids=[dataset.equilibrium_ids[row] for row in rows],
    )


def target_equilibria(
    dataset: dataset_types.RawDataset,
) -> list[desc_types.DescEquilibrium]:
    """Rebuild the DESC equilibrium each row of a dataset was solved as.

    The pressure comes from the stored spline rather than from the sampled `pressure`
    column: the solver saw the spline, so a force residual recomputed from the 20-point
    input would not reproduce the one the dataset records.

    Args:
        dataset: Dataset to rebuild the rows of.

    Returns:
        One equilibrium per row, in row order.
    """
    y = np.asarray(dataset.y)
    knots = np.asarray(dataset.pressure_knots)
    params = np.asarray(dataset.pressure_params)
    return [
        surrogate_utils.desc_equilibrium_from_prediction(
            surrogate_utils.unpack_flat_target(dataset.basis, y[row]),
            desc_types.DescSplineProfile(
                name="pressure", knots=knots[row], params=params[row]
            ),
        )
        for row in range(dataset.n_samples)
    ]


def assert_compatible_bases(
    left: surrogate_types.EquilibriumBasisSpecification,
    right: surrogate_types.EquilibriumBasisSpecification,
) -> None:
    """Raise unless two basis specifications agree entry by entry.

    A trained model, the dataset it was fitted on and an interpolator lattice only produce
    comparable `y` vectors while their bases match, and a mismatch is otherwise invisible:
    the vectors still have the right width whenever the block widths happen to coincide.

    Args:
        left: First basis specification.
        right: Second basis specification.
    """
    if left.resolution != right.resolution:
        raise ValueError(
            f"Incompatible resolutions: {left.resolution} against {right.resolution}."
        )
    for name in ("r_modes", "z_modes", "lambda_modes"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"Incompatible {name}.")
    for name in ("pressure_rho_grid", "iota_rho_grid"):
        if not np.allclose(getattr(left, name), getattr(right, name)):
            raise ValueError(f"Incompatible {name}.")


def _stack(
    frame: pd.DataFrame, column: str
) -> Float[NpOrJaxArray, "n_samples n_values"]:
    """Stack a parquet column of equal-length lists into a 2D array.

    Args:
        frame: Frame holding the column.
        column: Name of the column.

    Returns:
        The column as a float array with one row per sample.
    """
    return np.stack([np.asarray(row, dtype=float) for row in frame[column]], axis=0)
