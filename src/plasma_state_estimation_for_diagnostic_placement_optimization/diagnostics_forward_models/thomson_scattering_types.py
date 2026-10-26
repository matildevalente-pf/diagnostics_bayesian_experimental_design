from typing import Annotated

import jaxtyping as jt
import pydantic
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
    array_serializer,
    array_validator,
)


class ThomsonScatteringGeometry(pydantic.BaseModel):
    """Spatial definition of a Thomson scattering setup: where it looks.

    Explicit 3D positions in metres, independent of any equilibrium, plus the three
    numbers of the collection lens the etendue needs
    (`thomson_scattering_utils.collection_optics`). Build one with
    `geometry_utils.thomson_geometry_from_chord`.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    laser_origin: Annotated[
        jt.Float[NpOrJaxArray, "3"], array_validator, array_serializer
    ]
    """Beam entry point ``(x, y, z)``."""

    laser_direction: Annotated[
        jt.Float[NpOrJaxArray, "3"], array_validator, array_serializer
    ]
    """Unit beam direction ``(x, y, z)``."""

    reference_phi: ScalarFloat
    """Toroidal angle of the plane the chord lies in (used for the 2D plot)."""

    volume_positions: Annotated[
        jt.Float[NpOrJaxArray, "n_volume 3"], array_validator, array_serializer
    ]
    """Scattering-volume centres ``(x, y, z)``."""

    optic_positions: Annotated[
        jt.Float[NpOrJaxArray, "n_optic 3"], array_validator, array_serializer
    ]
    """Collection-lens positions ``(x, y, z)``."""

    volume_optic_index: Annotated[
        jt.Int[NpOrJaxArray, " n_volume"], array_validator, array_serializer
    ]
    """Index of the optic that reads each volume (one sightline, fiber and polychromator
    per volume)."""

    collection_aperture_diameter: ScalarFloat
    """Clear-aperture diameter of the collection lens in m; sets the solid angle."""

    collection_focal_length: ScalarFloat
    """Focal length of the collection lens in m; with the standoff it sets the
    magnification and hence the imaged volume length."""

    fiber_core_diameter: ScalarFloat
    """Diameter of the fiber bundle the lens images each volume onto, in m."""

    name: str = "thomson_scattering"
    """Human-readable label for the configuration."""


class PolychromatorResponse(pydantic.BaseModel):
    """Spectral response of a Thomson scattering system: the laser and the passbands.

    The forward-model counterpart to `ThomsonScatteringGeometry`. Wavelengths in m,
    energy in J.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    laser_wavelength: ScalarFloat
    """Laser wavelength (1064e-9 for the W7-X Nd:YAG)."""

    laser_pulse_energy: ScalarFloat
    """Laser pulse energy."""

    filter_centers: Annotated[
        jt.Float[NpOrJaxArray, " n_channel"], array_validator, array_serializer
    ]
    """Passband centre wavelengths (band midpoints)."""

    filter_widths: Annotated[
        jt.Float[NpOrJaxArray, " n_channel"], array_validator, array_serializer
    ]
    """Passband half-widths: each channel transmits ``[center - width, center + width]``
    with unit transmission and nothing outside."""

    band_scales: (
        Annotated[
            jt.Float[NpOrJaxArray, " n_volume"],
            array_validator,
            array_serializer,
        ]
        | None
    ) = None
    """Per-volume factor scaling every band edge's distance to the laser line
    (`thomson_scattering_utils.band_edges`); None applies the stored bands to every
    volume."""

    name: str = "polychromator"
    """Human-readable label for the response."""
