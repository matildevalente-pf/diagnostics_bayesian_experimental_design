# The pinned diagnostic design point, the instrument built from it, and the
# builder that turns one into the Thomson and interferometer geometries.

import hashlib
import math
from typing import Literal

import numpy as np
import pydantic
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    geometry_utils,
    interferometer_types,
    thomson_scattering_types,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
)

BandScaleRule = Literal["fisher_information", "none"]
"""Which per-volume band-scale rule a synthetic target applies; `none` keeps one."""

MACHINE_SCALE = surrogate_types.MachineScale(length_factor=5.5, field_factor=9.0)
"""The machine every truth is read at and the surrogate evaluated at: the dataset's
`R0 = 1.00 m`, `B = 1 T` boundary carried to `R0 = 5.5 m` and `B = 9 T`, so every
pressure is 81 times the dataset's and every flux 272.25 times."""

DEFAULT_CHORD_SURFACE_ANGLES = [
    [1.9382830959, 1.4074045709],
    [2.1034183528, 1.0139531101],
    [-3.1415926536, 0.0],
    [-2.1034183528, -1.0139531101],
    [-1.9382830959, -1.4074045709],
]
"""Interferometer reference design of Fig. 2.13(a) of the thesis."""

TE_REFERENCE_EV = 1.0e4
PRESSURE_REFERENCE_PA = 2.0e6
"""Anchor of the `sqrt(p)` split: 2 MPa, the median core of a 3 % beta truth at
`MACHINE_SCALE`, gets the 10 keV top of the W7-X Thomson design range (Bozhenkov
2017, section 2, https://doi.org/10.1088/1748-0221/12/10/P10004)."""


class DiagnosticDesignPoint(pydantic.BaseModel):
    """The pinned diagnostic layout, in angles on the auxiliary design surface.

    `build_diagnostic_geometry` turns it into the 3D diagnostic geometry.
    """

    toroidal_angle_deg: float = 0.0
    """Poloidal plane both diagnostics live in."""
    interferometer_toroidal_angle_deg: float | None = None
    """Poloidal plane of the interferometer chords when it is not `toroidal_angle_deg`,
    for layouts whose two diagnostics were placed independently; None shares it."""
    surface_offset: float = 0.14
    """Distance in m of the auxiliary design surface outside the plasma boundary
    (`geometry_utils.DEFAULT_SURFACE_OFFSET`), which fixes the surface the angles below
    are measured on."""

    laser_surface_angles: list[float] = [0.0, math.pi]
    """The two surface angles of the Thomson laser, entry first, in rad from `+R` about
    the cross-section centre.

    The default is the horizontal beam crossing at the height of the magnetic axis,
    entering on the low-field side and leaving on the high-field side, which is the
    direction the scattering angles are measured from.
    """
    delta: float = 0.0
    """Margin in m added outside each chord/boundary crossing before the volumes are
    placed.

    Zero keeps every volume inside the plasma, where the signal is: the laser's
    plasma path is 0.52 m at `MACHINE_SCALE`, and a volume placed in the vacuum
    outside carries no information at any noise model.
    """
    volume_span_fractions: list[float] = list(
        geometry_utils.uniform_volume_fractions(15)
    )
    """Where each scattering volume sits along the laser's plasma span, as a fraction
    from the entry (0) to the exit (1), one polychromator per volume.

    Fifteen equal cells by default; an empty list is a layout without Thomson
    scattering.
    """

    optic_aim_fractions: list[float] | None = None
    """Span positions whose centroid on each side of the axis a lens aims at; None aims
    it at the volumes it serves.

    Set to another layout's, it keeps that layout's lenses in place whatever volumes
    this one puts on the beam.
    """

    optic_surface_angles: list[float] = [
        -150.0 * math.pi / 180.0,
        20.0 * math.pi / 180.0,
    ]
    """The surface angles the two lenses look through, forward-scatter set first.

    It serves
    the high-field-side volumes at 34.9-38.7 deg of scattering angle, inside
    `geometry_utils.LOW_ANGLE_RANGE_DEG`; the back-scatter set serves the low-field-side
    ones at 146.9-150.7 deg, inside `HIGH_ANGLE_RANGE_DEG`. Both windows move with the
    laser.
    """
    band_scale_rule: BandScaleRule = "fisher_information"
    """Per-volume band scale at the state a reference equilibrium predicts there, by
    `thomson_scattering_utils.fisher_band_scale`, the
    counterpart of the filter cut-off design of Dreier 2006 section II.B,
    https://doi.org/10.13182/FST06-A1244."""

    thomson_noise_model: thomson_scattering_utils.ThomsonNoiseModel = "electronic"
    """Which uncertainty every Thomson reading carries
    (`thomson_scattering_utils.signal_uncertainties`): the photon-counting limit
    `sqrt(D_i)` or the W7-X detector noise curve.

    A property of the instrument, so it enters the recorded data and its uncertainty
    alike.
    """

    band_scale: float = 1.0
    """Factor scaling every band edge's distance to the laser line
    (`thomson_scattering_utils.band_edges`), used when `band_scale_rule` is `none`.

    1.0
    is the W7-X band set; the admissible interval is
    `thomson_scattering_utils.band_scale_bounds`.
    """

    chord_surface_angles: list[list[float]] = DEFAULT_CHORD_SURFACE_ANGLES
    """The two surface angles of each interferometer chord, in rad, one `[start, end]`
    pair per line of sight and the variables the placement optimization moves.

    Pinned to `DEFAULT_CHORD_SURFACE_ANGLES` so the layout does not move with the
    equilibrium; an empty list is a layout without an interferometer.
    """

    @pydantic.model_serializer(mode="wrap")
    def _omit_unset_optional_fields(
        self, handler: pydantic.SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        """Leave `optic_aim_fractions` and `interferometer_toroidal_angle_deg` out while
        unset, so a placement keeps the content hash it had before either existed."""
        data = handler(self)
        for name in ("optic_aim_fractions", "interferometer_toroidal_angle_deg"):
            if getattr(self, name) is None:
                data.pop(name, None)
        return data

    @pydantic.model_validator(mode="before")
    @classmethod
    def _read_legacy_field_names(cls, data: object) -> object:
        """Placements stored before 2026-10-09 spell `surface` as `wall` in the field
        names, and are read under the current ones."""
        if not isinstance(data, dict):
            return data
        return {key.replace("wall", "surface"): value for key, value in data.items()}


class DiagnosticPlacement(pydantic.BaseModel):
    """The built instrument every truth is measured with, and what it was built from."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    design: DiagnosticDesignPoint
    """The design point the layout, calibration and noise model come from."""

    reference_equilibrium_id: str
    """Equilibrium the volumes and chords were placed on and the band scales read."""

    thomson_geometry: thomson_scattering_types.ThomsonScatteringGeometry | None
    """Scattering volumes and collection optics; None for a design without them."""

    polychromator_response: thomson_scattering_types.PolychromatorResponse
    """Laser and passbands, with one band scale per volume if the rule is on."""

    thomson_signal_scale: float
    """`kappa` in V s per photoelectron, converting `D_i` to the integrated detector
    signal the electronic noise curve is written in
    (`thomson_scattering_utils.electronic_signal_scale`).

    Calibrated once on the reference configuration and the same for every layout.
    """

    interferometer_chords: list[interferometer_types.InterferometerGeometry]
    """Lines of sight of the interferometer fan, design surface to design surface."""

    @property
    def placement_id(self) -> str:
        """Content hash of the placement, so equal instruments share one id."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def build_diagnostic_geometry(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    design: DiagnosticDesignPoint | None = None,
    name: str = "layout",
) -> tuple[
    thomson_scattering_types.ThomsonScatteringGeometry | None,
    thomson_scattering_types.PolychromatorResponse,
    list[interferometer_types.InterferometerGeometry],
]:
    """The Thomson + interferometer layout of a `DiagnosticDesignPoint`.

    One poloidal plane holds both diagnostics unless the design point gives the chords
    their own, and every angle of the design point is read on its plane's auxiliary
    design surface, so the laser, the chords and the optic ports are placed the same way
    (`geometry_utils.surface_point_at_angle`).

    Args:
        coefficients: DESC Fourier-Zernike geometry coefficients of the equilibrium.
        design: The design-point parameters; the pinned defaults when `None`.
        name: Prefix of the geometry names.

    Returns:
        The Thomson geometry (None for a design without volumes), the polychromator
        response, and the interferometer chords.
    """
    design = design or DiagnosticDesignPoint()
    toroidal_angle = np.radians(design.toroidal_angle_deg)
    laser_surface_angles = (
        design.laser_surface_angles[0],
        design.laser_surface_angles[1],
    )
    geometry = (
        geometry_utils.thomson_geometry_from_chord(
            coefficients,
            toroidal_angle=toroidal_angle,
            surface_offset=design.surface_offset,
            laser_surface_angles=laser_surface_angles,
            delta=design.delta,
            volume_fractions=tuple(design.volume_span_fractions),
            optic_surface_angles=(
                design.optic_surface_angles[0],
                design.optic_surface_angles[1],
            ),
            name=f"{name}_thomson",
            aim_fractions=(
                None
                if design.optic_aim_fractions is None
                else tuple(design.optic_aim_fractions)
            ),
        )
        if design.volume_span_fractions
        else None
    )
    response = thomson_scattering_utils.build_polychromator_response(
        polychromator_bands=list(thomson_scattering_utils.W7X_POLYCHROMATOR_BANDS),
        laser_wavelength=thomson_scattering_utils.W7X_LASER_WAVELENGTH,
        laser_pulse_energy=thomson_scattering_utils.W7X_LASER_PULSE_ENERGY,
    )
    if (
        geometry is not None
        and design.band_scale_rule == "none"
        and design.band_scale != 1.0
    ):
        n_volume = np.asarray(geometry.volume_positions).shape[0]
        response = response.model_copy(
            update={"band_scales": np.full(n_volume, design.band_scale)}
        )
    chord_toroidal_angle = (
        toroidal_angle
        if design.interferometer_toroidal_angle_deg is None
        else np.radians(design.interferometer_toroidal_angle_deg)
    )
    _, surface = geometry_utils.cross_section_and_surface(
        coefficients, chord_toroidal_angle, design.surface_offset
    )
    chords = geometry_utils.chords_from_surface_angles(
        surface,
        np.asarray(design.chord_surface_angles, dtype=float).reshape(-1, 2),
        name_prefix=f"{name}_chord",
    )
    return geometry, response, chords
