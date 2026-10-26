from typing import Annotated, Literal, Self

import pydantic
from jaxtyping import Float, Int
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
    array_serializer,
    array_validator,
)


class DescResolution(pydantic.BaseModel):
    """Spectral resolution and symmetry defining a DESC Fourier-Zernike basis.

    `DescEquilibrium` stores its geometry as coefficients plus explicit `(l, m, n)`
    mode tables, so it never needs this to be interpreted. Rebuilding a live
    `desc.equilibrium.Equilibrium` from it does: the DESC constructor takes the
    resolution rather than the mode tables. The values cannot be recovered from a
    mode table in general, because under `"fringe"` indexing the largest radial mode
    number exceeds `radial_resolution`.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    radial_resolution: Annotated[int, pydantic.Field(ge=0)]
    """Maximum radial (Zernike) mode number `L`."""

    poloidal_resolution: Annotated[int, pydantic.Field(ge=0)]
    """Maximum poloidal mode number `M`."""

    toroidal_resolution: Annotated[int, pydantic.Field(ge=0)]
    """Maximum toroidal mode number `N`, in per-field-period units."""

    n_field_periods: Annotated[int, pydantic.Field(ge=1)]
    """Number of toroidal field periods."""

    is_stellarator_symmetric: bool = True
    """Whether the basis is restricted to stellarator-symmetric modes."""

    spectral_indexing: Literal["ansi", "fringe"] = "ansi"
    """Rule relating the radial and poloidal mode numbers of the Zernike basis."""


class DescFourierZernikeCoefficients(pydantic.BaseModel):
    """Minimal DESC Fourier-Zernike geometry coefficient set.

    Self-contained subset of a DESC equilibrium needed to reconstruct the
    flux-surface geometry: the spectral coefficients together with their explicit
    `(l, m, n)` mode tables, so no DESC basis reconstruction is required to
    interpret them.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    r_lmn: Float[NpOrJaxArray, " n_modes_r"]
    """Fourier-Zernike coefficients of the cylindrical coordinate `R` [m]."""

    z_lmn: Float[NpOrJaxArray, " n_modes_z"]
    """Fourier-Zernike coefficients of the cylindrical coordinate `Z` [m]."""

    lambda_lmn: Float[NpOrJaxArray, " n_modes_lambda"]
    """Fourier-Zernike coefficients of the poloidal stream function `lambda` [rad]."""

    r_modes: Int[NpOrJaxArray, "n_modes_r 3"]
    """`(l, m, n)` mode numbers of `r_lmn`; `n` is in per-field-period units."""

    z_modes: Int[NpOrJaxArray, "n_modes_z 3"]
    """`(l, m, n)` mode numbers of `z_lmn`; `n` is in per-field-period units."""

    lambda_modes: Int[NpOrJaxArray, "n_modes_lambda 3"]
    """`(l, m, n)` mode numbers of `lambda_lmn`; `n` is in per-field-period units."""

    n_field_periods: int
    """Number of toroidal field periods."""

    @pydantic.model_validator(mode="after")
    def _check_modes_match_coefficients(self) -> Self:
        for coefficients, modes, name in (
            (self.r_lmn, self.r_modes, "r"),
            (self.z_lmn, self.z_modes, "z"),
            (self.lambda_lmn, self.lambda_modes, "lambda"),
        ):
            if coefficients.shape[0] != modes.shape[0]:
                raise ValueError(
                    f"{name}_lmn has {coefficients.shape[0]} coefficients but "
                    f"{name}_modes has {modes.shape[0]} rows."
                )
        return self


class DescSplineProfile(pydantic.BaseModel):
    """Serialized representation of a DESC `desc.profiles.SplineProfile` object."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    method: str = "cubic2"
    """Spline interpolation method of the profile (e.g. 'linear', 'cubic', 'cubic2')."""

    name: str = ""
    """Name of the profile (e.g. 'pressure', 'current', 'iota')."""

    knots: Annotated[Float[NpOrJaxArray, " n_knots"], array_validator, array_serializer]
    """Knots of the spline profile."""

    params: Annotated[Float[NpOrJaxArray, " n_knots"], array_validator, array_serializer]
    """Parameters of the spline profile."""


class DescEquilibrium(pydantic.BaseModel):
    """A DESC equilibrium: flux-surface geometry plus its 1D profiles.

    The geometry lives in an embedded `DescFourierZernikeCoefficients` (explicit mode
    tables, so no DESC basis reconstruction is needed). A pressure profile is required
    and exactly one of `iota_profile` / `current_profile` must be set.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    coefficients: DescFourierZernikeCoefficients
    """Fourier-Zernike geometry coefficients with their `(l, m, n)` mode tables."""

    total_toroidal_flux: ScalarFloat = 1.0
    """Total toroidal flux (in Webers) within the last closed flux surface."""

    pressure_profile: DescSplineProfile
    """Pressure profile, in Pa over `rho`."""

    iota_profile: DescSplineProfile | None = None
    """Rotational-transform profile over `rho`."""

    current_profile: DescSplineProfile | None = None
    """Toroidal-current profile over `rho`."""

    @property
    def n_field_periods(self) -> int:
        """Number of toroidal field periods of the geometry."""
        return self.coefficients.n_field_periods

    @pydantic.model_validator(mode="after")
    def _check_that_either_iota_or_current_are_provided(self) -> Self:
        if self.iota_profile is None and self.current_profile is None:
            raise ValueError(
                "Either the iota_profile or current_profile must be provided."
            )
        if self.iota_profile is not None and self.current_profile is not None:
            raise ValueError(
                "Only one of iota_profile or current_profile should be provided."
            )
        return self
