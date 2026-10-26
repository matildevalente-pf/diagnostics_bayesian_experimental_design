from typing import NamedTuple, Self

import numpy as np
import pydantic
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
)


class PreprocessorParams(pydantic.BaseModel):
    """Fitted input and output normalization, with optional output PCA."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    x_mean: Float[NpOrJaxArray, " n_rho"]
    """Mean of each pressure sample point over the training rows [Pa]."""

    x_std: Float[NpOrJaxArray, " n_rho"]
    """Standard deviation of each pressure sample point, with zeros replaced by one."""

    y_energy_scale: Float[NpOrJaxArray, " n_y"]
    """Per-entry weight the target is divided by before z-scoring."""

    y_mean: Float[NpOrJaxArray, " n_y"]
    """Mean of the energy-scaled target over the training rows."""

    y_std: Float[NpOrJaxArray, " n_y"]
    """Standard deviation of the energy-scaled target, with zeros replaced by one."""

    y_pca_components: Float[NpOrJaxArray, "n_components n_y"] | None = None
    """PCA eigenvectors as rows, or None when the network predicts `y` directly."""

    y_pca_mean: Float[NpOrJaxArray, " n_y"] | None = None
    """Centering vector of the PCA, in normalized target space."""

    @property
    def n_latent(self) -> int:
        """Width of the space the network predicts in: the PCA rank, or `n_y`."""
        if self.y_pca_components is None:
            return int(self.y_mean.shape[0])
        return int(self.y_pca_components.shape[0])

    @pydantic.model_validator(mode="after")
    def _check_shapes_agree(self) -> Self:
        n_y = self.y_mean.shape[0]
        for array, name in (
            (self.y_std, "y_std"),
            (self.y_energy_scale, "y_energy_scale"),
        ):
            if array.shape[0] != n_y:
                raise ValueError(
                    f"{name} is {array.shape[0]} wide but y_mean is {n_y}."
                )
        if self.x_mean.shape != self.x_std.shape:
            raise ValueError(
                f"x_mean is {self.x_mean.shape} but x_std is {self.x_std.shape}."
            )
        if (self.y_pca_components is None) != (self.y_pca_mean is None):
            raise ValueError(
                "y_pca_components and y_pca_mean must be given together or not at all."
            )
        if self.y_pca_components is not None and self.y_pca_components.shape[1] != n_y:
            raise ValueError(
                f"y_pca_components acts on {self.y_pca_components.shape[1]} entries "
                f"but y_mean is {n_y} wide."
            )
        if np.any(np.asarray(self.y_energy_scale) <= 0.0):
            raise ValueError("y_energy_scale must be positive the target divides by it")
        return self


class PressureParams(NamedTuple):
    """Parameters of the analytic pressure profile `p(rho) = p0 * (1 - rho^a)^b`."""

    p0: ScalarFloat
    """On-axis pressure [Pa]."""

    a: ScalarFloat
    """Exponent setting how far out the profile stays flat."""

    b: ScalarFloat
    """Exponent setting how steeply it falls; below 1 the edge gradient diverges."""
