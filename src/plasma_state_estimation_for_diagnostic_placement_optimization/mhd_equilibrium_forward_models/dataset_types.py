# Containers for a surrogate training dataset: the sidecar settings JSON, the arrays
# read out of the parquet, and the train/validation/test partition.
#
# The split's id lists are row indices, resolved once at load time against the id
# column, and the basis fields form a single `EquilibriumBasisSpecification`, so a
# dataset, a model and an interpolator lattice can be compared field by field.

from typing import Self

import numpy as np
import pydantic
from jaxtyping import Float, Int
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
)


class DatasetMetadata(pydantic.BaseModel):
    """Contents of a dataset's sidecar settings JSON, minus the parquet itself.

    The settings file is the tracked half of a dataset: the parquet is too large for git
    and is regenerated, while this carries the basis, the provenance and the pinned split
    that make two training runs comparable.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    """Dataset name, matching the parquet and settings file stem."""

    generated: str
    """Date the settings file was written, as `YYYY-MM-DD`."""

    plasma_configuration_id: str
    """Id of the plasma boundary every equilibrium in the dataset was solved on.

    Provenance only; nothing in this package resolves it.
    """

    recommended_max_fnorm_vol: float | None = None
    """Force-residual threshold above which a solve is considered unconverged.

    The dataset is exported unfiltered, so this is the filter a training run is expected
    to apply, and the one the pinned split was defined under. None for a dataset with no
    recommendation, such as the interpolator lattice, whose nodes all converge.
    """

    n_samples: int
    """Number of rows in the parquet, before any filtering."""

    basis: surrogate_types.EquilibriumBasisSpecification
    """Spectral basis and radial grids the dataset's `y` vectors live in."""

    train_ids: list[str] | None = None
    """Equilibrium ids of the training partition, or None for a dataset with no split."""

    validation_ids: list[str] | None = None
    """Equilibrium ids of the validation partition."""

    test_ids: list[str] | None = None
    """Equilibrium ids of the test partition."""

    @property
    def has_split(self) -> bool:
        """Whether the settings file pins a train/validation/test partition."""
        return self.train_ids is not None

    @pydantic.model_validator(mode="after")
    def _check_split_is_complete_and_disjoint(self) -> Self:
        partitions = (self.train_ids, self.validation_ids, self.test_ids)
        if any(part is None for part in partitions) and not all(
            part is None for part in partitions
        ):
            raise ValueError(
                "train_ids, validation_ids and test_ids must be given together or "
                "not at all."
            )
        if self.train_ids is None:
            return self
        sizes = sum(len(part) for part in partitions if part is not None)
        union = set().union(*(set(part) for part in partitions if part is not None))
        if len(union) != sizes:
            raise ValueError(
                f"The split partitions overlap: {sizes} ids but {len(union)} distinct."
            )
        return self


class RawDataset(pydantic.BaseModel):
    """One dataset parquet, read into flat arrays in its stored row order.

    All arrays are float64 and share a leading sample axis. `y` is the flat target every
    surrogate predicts, laid out as `surrogate_types.FLAT_TARGET_LAYOUT`.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    basis: surrogate_types.EquilibriumBasisSpecification
    """Spectral basis and radial grids the arrays below are expressed in."""

    pressure: Float[NpOrJaxArray, "n_samples n_rho"]
    """Pressure on `basis.pressure_rho_grid`, the model input [Pa]."""

    y: Float[NpOrJaxArray, "n_samples n_y"]
    """Flat target vectors, one row per sample."""

    pressure_knots: Float[NpOrJaxArray, "n_samples n_knots"]
    """Knots in `rho` of the pressure spline each equilibrium was actually solved with."""

    pressure_params: Float[NpOrJaxArray, "n_samples n_knots"]
    """Spline values at `pressure_knots` [Pa].

    Kept alongside the sampled `pressure` because a force-balance residual recomputed
    from the 20-point input does not reproduce the solver's own, while one recomputed
    from the stored spline does.
    """

    fnorm_vol: Float[NpOrJaxArray, " n_samples"]
    """Volume-averaged normalized force residual reported by the DESC solve."""

    beta_vol: Float[NpOrJaxArray, " n_samples"]
    """Volume-averaged plasma beta of the solved equilibrium."""

    equilibrium_ids: list[str]
    """Id of each row's solved equilibrium, in row order; provenance only."""

    @property
    def n_samples(self) -> int:
        """Number of rows."""
        return int(self.pressure.shape[0])

    @pydantic.model_validator(mode="after")
    def _check_shapes_agree(self) -> Self:
        n_samples = self.pressure.shape[0]
        for array, name in (
            (self.y, "y"),
            (self.pressure_knots, "pressure_knots"),
            (self.pressure_params, "pressure_params"),
            (self.fnorm_vol, "fnorm_vol"),
            (self.beta_vol, "beta_vol"),
        ):
            if array.shape[0] != n_samples:
                raise ValueError(
                    f"{name} has {array.shape[0]} rows but pressure has {n_samples}."
                )
        if len(self.equilibrium_ids) != n_samples:
            raise ValueError(
                f"equilibrium_ids has {len(self.equilibrium_ids)} entries but pressure "
                f"has {n_samples} rows."
            )
        if self.pressure.shape[1] != self.basis.n_rho:
            raise ValueError(
                f"pressure is sampled at {self.pressure.shape[1]} points but the basis "
                f"grid has {self.basis.n_rho}."
            )
        if self.y.shape[1] != self.basis.n_y:
            raise ValueError(
                f"y is {self.y.shape[1]} wide but the basis implies {self.basis.n_y}."
            )
        if self.pressure_knots.shape[1] != self.pressure_params.shape[1]:
            raise ValueError(
                f"pressure_knots is {self.pressure_knots.shape[1]} wide but "
                f"pressure_params is {self.pressure_params.shape[1]}."
            )
        return self


class DataSplit(pydantic.BaseModel):
    """Row indices of the train, validation and test partitions of a `RawDataset`.

    Indices rather than ids, because the partitions are resolved once against the loaded
    dataset and then used to slice arrays.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    train: Int[NpOrJaxArray, " n_train"]
    """Row indices of the training partition."""

    validation: Int[NpOrJaxArray, " n_validation"]
    """Row indices of the validation partition."""

    test: Int[NpOrJaxArray, " n_test"]
    """Row indices of the test partition."""

    @pydantic.model_validator(mode="after")
    def _check_partitions_are_disjoint(self) -> Self:
        indices = np.concatenate(
            [np.asarray(self.train), np.asarray(self.validation), np.asarray(self.test)]
        )
        if len(np.unique(indices)) != len(indices):
            raise ValueError("The split partitions share at least one row index.")
        return self
