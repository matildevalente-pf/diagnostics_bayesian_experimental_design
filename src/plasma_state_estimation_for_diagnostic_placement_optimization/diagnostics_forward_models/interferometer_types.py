from typing import Annotated

import jaxtyping as jt
import pydantic

from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    array_serializer,
    array_validator,
)


class InterferometerGeometry(pydantic.BaseModel):
    """Line of sight of a single interferometer chord: where one channel looks.

    A chord is the straight beam between an entry and an exit point; the channel
    measures the electron density integrated along it. The geometry is stored as
    explicit lab-frame 3D endpoints in metres, so it is independent of any
    equilibrium (an equilibrium is needed only to supply the local `ne` along the
    line, or to draw flux surfaces behind it). It is the interferometer
    counterpart of `ThomsonScatteringGeometry`. One object per line of sight.

    Conventions: lengths in m.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    chord_start: Annotated[jt.Float[NpOrJaxArray, "3"], array_validator, array_serializer]
    """Chord entry point ``(x, y, z)``."""

    chord_end: Annotated[jt.Float[NpOrJaxArray, "3"], array_validator, array_serializer]
    """Chord exit point ``(x, y, z)``."""

    name: str = "interferometer_chord"
    """Human-readable label for the line of sight."""
