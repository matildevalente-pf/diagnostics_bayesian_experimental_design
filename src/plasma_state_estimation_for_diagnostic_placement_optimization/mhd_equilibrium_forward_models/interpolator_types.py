from typing import Self

import numpy as np
import pydantic
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
)


class InterpolatorLattice(pydantic.BaseModel):
    """Solved equilibria on a regular lattice in the pressure parameters `(p0, a, b)`.

    The interpolator projects any pressure profile onto `p(rho) = p0 (1 - rho^a)^b` and
    interpolates `y` trilinearly between the eight surrounding nodes.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    basis: surrogate_types.EquilibriumBasisSpecification
    """Spectral basis and radial grids the lattice's `y` vectors live in."""

    p0_grid: Float[NpOrJaxArray, " n_p0"]
    """On-axis pressures of the lattice, increasing [Pa]."""

    a_grid: Float[NpOrJaxArray, " n_a"]
    """Width exponents of the lattice, increasing."""

    b_grid: Float[NpOrJaxArray, " n_b"]
    """Steepness exponents of the lattice, increasing."""

    y_lattice: Float[NpOrJaxArray, "n_p0 n_a n_b n_y"]
    """Flat target vector solved at every lattice node."""

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of nodes along each of the three parameter axes."""
        return (
            int(self.p0_grid.shape[0]),
            int(self.a_grid.shape[0]),
            int(self.b_grid.shape[0]),
        )

    @pydantic.model_validator(mode="after")
    def _check_grids_and_lattice_agree(self) -> Self:
        for grid, name in (
            (self.p0_grid, "p0_grid"),
            (self.a_grid, "a_grid"),
            (self.b_grid, "b_grid"),
        ):
            values = np.asarray(grid)
            if values.shape[0] < 2:
                raise ValueError(f"{name} needs at least two nodes to interpolate.")
            if np.any(np.diff(values) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing.")
        expected = (*self.shape, self.basis.n_y)
        if tuple(self.y_lattice.shape) != expected:
            raise ValueError(
                f"y_lattice has shape {tuple(self.y_lattice.shape)} but the grids and "
                f"the basis imply {expected}."
            )
        return self
