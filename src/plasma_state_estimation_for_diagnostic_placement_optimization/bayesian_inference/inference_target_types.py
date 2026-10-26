# The observed data one inference run compares its forward model against.
#
# The likelihood compares raw polychromator signals `D_i` and interferometer
# line-integrated densities and nothing else. An inference sees only these comparison
# vectors, their 1-sigma uncertainties, and the geometry it has to forward-model
# through.
#
# Invalid channels are marked by a non-positive or non-finite `sigma`, which is how
# `likelihood_utils` masks them, and how the faint-channel mask of
# `synthetic_target_utils` records a dropped channel.
#
# Conventions: SI (`ne` in m^-3, `Te` in eV, positions in m, `D_i` in photoelectrons).

from typing import Annotated

import jaxtyping as jt
import pydantic
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    array_serializer,
    array_validator,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    interferometer_types,
    thomson_scattering_types,
)


class ThomsonSignalTarget(pydantic.BaseModel):
    """Recorded Thomson per-channel signals `D_i`, plus the instrument that made them.

    Unlike a target of inverted `ne`/`Te`, this compares the raw polychromator signals,
    so the likelihood has to forward-model a guess through the same geometry and
    response the data came from; both are therefore carried here.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    geometry: thomson_scattering_types.ThomsonScatteringGeometry
    """Scattering volumes and collection optics the signals were recorded with."""

    response: thomson_scattering_types.PolychromatorResponse
    """Laser and filter-bank properties of the polychromator."""

    signals: Annotated[
        jt.Float[NpOrJaxArray, "n_volume n_channel"], array_validator, array_serializer
    ]
    """Recorded signal of every channel of every volume, in photoelectrons."""

    uncertainties: Annotated[
        jt.Float[NpOrJaxArray, "n_volume n_channel"], array_validator, array_serializer
    ]
    """1-sigma uncertainty of each signal; zero marks a masked channel.

    Whichever of the two models of `thomson_scattering_utils.signal_uncertainties`
    the instrument carries. This is the whole covariance the likelihood uses, the error
    shared within a volume being recorded below but not modelled.
    """

    global_calibration: Annotated[
        jt.Float[NpOrJaxArray, " n_volume"], array_validator, array_serializer
    ]
    """Nominal geometric absolute calibration `dL dOmega` of every volume
    (`thomson_scattering_utils.absolute_calibration`), which the likelihood
    forward-models with, times that volume's inferred correction when one is
    inferred."""

    true_calibration_factors: (
        Annotated[
            jt.Float[NpOrJaxArray, " n_volume"], array_validator, array_serializer
        ]
        | None
    ) = None
    """Corrections `delta_j` the data were generated with, None meaning the nominal
    geometric calibration on every volume; for coverage checks only, the likelihood
    never reads it."""


class InterferometerTarget(pydantic.BaseModel):
    """Recorded line-integrated electron density per chord, with the chord geometries.

    The geometries are carried so the likelihood can integrate a guess along the same
    lines of sight. They are the trimmed, in-plasma segments when
    `PlasmaEstimationSettings.trim_chords_to_plasma` is on, because the truth and the
    inference have to sample identically.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    geometries: Annotated[
        list[interferometer_types.InterferometerGeometry], pydantic.Field(min_length=1)
    ]
    """One line of sight per chord, in the order of the two arrays below."""

    line_integrated_density: Annotated[
        jt.Float[NpOrJaxArray, " n_chord"], array_validator, array_serializer
    ]
    """Recorded line-integrated density of each chord, in m^-2."""

    line_integrated_density_sigma: Annotated[
        jt.Float[NpOrJaxArray, " n_chord"], array_validator, array_serializer
    ]
    """1-sigma uncertainty of each chord; zero marks a chord the likelihood drops."""


class InferenceTarget(pydantic.BaseModel):
    """Everything one inference run is fitted against.

    One optional sub-object per diagnostic, so a placement study can switch either off
    and the likelihood sums whichever are present. With both set to None the likelihood
    is identically zero and the run samples the prior, which is a legitimate
    configuration rather than an error.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    truth_name: str = ""
    """Which ground truth the synthetic data was generated from, for provenance."""

    thomson_signals: ThomsonSignalTarget | None = None
    """Thomson polychromator signals, or None to leave Thomson out."""

    interferometer: InterferometerTarget | None = None
    """Interferometer line integrals, or None to leave the interferometer out."""
