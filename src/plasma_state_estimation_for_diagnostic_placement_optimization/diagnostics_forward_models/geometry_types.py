# The surface a diagnostic layout places its angles on, shared by both instruments.

from typing import NamedTuple

import jaxtyping as jt
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)


class PoloidalSurface(NamedTuple):
    """A closed curve in one poloidal plane that a layout places its angles on.

    Every line of sight and every optic port is fixed by angles on one of these
    (`geometry_utils.surface_point_at_angle`), so the layout is a list of angles.
    """

    curve: jt.Float[NpOrJaxArray, "2 n_point"]
    """Closed `(R, Z)` polygon, star-shaped about `centre`."""

    centre: jt.Float[NpOrJaxArray, " 2"]
    """`(R, Z)` the angles are measured from, anticlockwise from `+R`."""

    toroidal_angle: float
    """Poloidal plane the surface lies in, in rad."""
