# The vocabulary every equilibrium surrogate in this package shares: the spectral basis
# and radial grids that one dataset, one trained model and one interpolator lattice all
# have to agree on.

from typing import Annotated, Self

import numpy as np
import pydantic
from jaxtyping import Float, Int
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)

FLAT_TARGET_LAYOUT = "y = [R_lmn | Z_lmn | lambda_lmn | iotaf | Psi]"
"""Layout of the flat target vector every surrogate predicts.

The blocks are concatenated in this order, so `y` has `n_r + n_z + n_lambda + n_iota+1`
entries. `Psi` is a single scalar rather than a profile because the toroidal flux
profile is exactly `phi(s) = Psi * s`.
"""


class MachineScale(pydantic.BaseModel):
    """Length and field factors carrying a normalized equilibrium to a machine.

    Ideal MHD is invariant under `R, Z -> L R, L Z` with `Psi -> L^2 Psi` and under
    `Psi -> b Psi` with `p -> b^2 p`; `lambda` and `iota` are dimensionless.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    length_factor: Annotated[float, pydantic.Field(gt=0.0)] = 1.0
    """`L`, multiplying every `R` and `Z` coefficient."""

    field_factor: Annotated[float, pydantic.Field(gt=0.0)] = 1.0
    """`b`, multiplying the magnetic field."""

    @property
    def pressure_factor(self) -> float:
        """`b^2`, carrying a normalized pressure to the machine."""
        return self.field_factor**2

    @property
    def flux_factor(self) -> float:
        """`b L^2`, carrying the toroidal flux to the machine."""
        return self.field_factor * self.length_factor**2


class EquilibriumBasisSpecification(pydantic.BaseModel):
    """The spectral basis and radial grids one surrogate dataset is expressed in.

    Two artifacts are comparable only if their basis specifications match: the flat
    target vector `y` is a concatenation of blocks whose meaning is fixed by the mode
    tables and by the iota grid, and the model input is a pressure profile sampled on
    `pressure_rho_grid`.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    resolution: desc_types.DescResolution
    """Spectral resolution, field-period count, symmetry and indexing of the basis."""

    r_modes: Int[NpOrJaxArray, "n_modes_r 3"]
    """`(l, m, n)` mode numbers of the `R_lmn` block of `y`."""

    z_modes: Int[NpOrJaxArray, "n_modes_z 3"]
    """`(l, m, n)` mode numbers of the `Z_lmn` block of `y`."""

    lambda_modes: Int[NpOrJaxArray, "n_modes_lambda 3"]
    """`(l, m, n)` mode numbers of the `lambda_lmn` block of `y`."""

    pressure_rho_grid: Float[NpOrJaxArray, " n_rho"]
    """Radial points the pressure profile, i.e. the model input, is sampled on [Pa]."""

    iota_rho_grid: Float[NpOrJaxArray, " n_iota"]
    """Radial points the `iotaf` block of `y` is sampled on.

    VMEC's full-grid convention, `rho = sqrt(linspace(0, 1, n_iota))`, so that the block
    is uniform in the normalized toroidal flux `s`.
    """

    @property
    def n_r(self) -> int:
        """Width of the `R_lmn` block of `y`."""
        return int(self.r_modes.shape[0])

    @property
    def n_z(self) -> int:
        """Width of the `Z_lmn` block of `y`."""
        return int(self.z_modes.shape[0])

    @property
    def n_lambda(self) -> int:
        """Width of the `lambda_lmn` block of `y`."""
        return int(self.lambda_modes.shape[0])

    @property
    def n_iota(self) -> int:
        """Width of the `iotaf` block of `y`."""
        return int(self.iota_rho_grid.shape[0])

    @property
    def n_rho(self) -> int:
        """Width of the model input, i.e. the number of pressure samples."""
        return int(self.pressure_rho_grid.shape[0])

    @property
    def n_y(self) -> int:
        """Width of the flat target vector `y`, including the trailing `Psi` scalar."""
        return self.n_r + self.n_z + self.n_lambda + self.n_iota + 1

    @pydantic.model_validator(mode="after")
    def _check_grids_are_usable(self) -> Self:
        for grid, name in (
            (self.pressure_rho_grid, "pressure_rho_grid"),
            (self.iota_rho_grid, "iota_rho_grid"),
        ):
            values = np.asarray(grid)
            if values.shape[0] < 2:
                raise ValueError(f"{name} needs at least two points.")
            if np.any(np.diff(values) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing.")
            if values[0] < 0.0 or values[-1] > 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        return self


class EquilibriumPrediction(pydantic.BaseModel):
    """One flat target vector split back into the quantities it stands for.

    Produced both by a surrogate forward pass and by unpacking a stored training target,
    so a prediction and its target are the same type and can be compared directly.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    coefficients: desc_types.DescFourierZernikeCoefficients
    """Fourier-Zernike geometry, i.e. the `R_lmn`, `Z_lmn` and `lambda_lmn` blocks."""

    iotaf: Float[NpOrJaxArray, " n_iota"]
    """Rotational transform on the basis' `iota_rho_grid`, in DESC's sign convention."""

    total_toroidal_flux: ScalarFloat
    """Toroidal flux through the last closed flux surface `Psi` [Wb].

    The flux profile is exactly `phi(s) = Psi * s`, so the scalar is the whole profile.
    """
