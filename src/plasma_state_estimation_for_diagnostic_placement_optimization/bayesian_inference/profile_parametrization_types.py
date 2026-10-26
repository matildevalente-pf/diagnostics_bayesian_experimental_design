# The `ne`/`Te` radial profile parametrizations the inference fits over, and their
# evaluation: SI (`ne` in m^-3, `Te` in eV) on `rho = rho_tor_norm` in `[0, 1]`.

from typing import Annotated, Literal, Self

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pydantic
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    array_serializer,
    array_validator,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_utils,
)

ProfileFamily = Literal["two_power_zero_edge", "spline_knots"]
"""Which parametrization a `PlasmaProfiles` carries; each formula is on its evaluator
below. `two_power_zero_edge` is the inferable one; the ground truths are built in
`spline_knots`."""

TWO_POWER_ZERO_EDGE_PARAM_NAMES: tuple[str, ...] = ("center", "alpha", "beta")
"""Order of the `two_power_zero_edge` parameter vector."""


class PlasmaProfiles(pydantic.BaseModel):
    """`ne` and `Te` radial profiles sharing one parametrization family.

    The family fixes the length and the meaning of both parameter vectors, which the
    validator enforces; the fields below say what each of them holds.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    family: ProfileFamily
    """Which parametrization both parameter vectors are expressed in."""

    electron_density_params: Annotated[
        jt.Float[NpOrJaxArray, " n_ne"], array_validator, array_serializer
    ]
    """`two_power_zero_edge`: `[center, alpha, beta]`, `center` in units of `1e19 m^-3`.
    `spline_knots`: `log(ne)` of the spline component at each knot."""

    electron_temperature_params: Annotated[
        jt.Float[NpOrJaxArray, " n_te"], array_validator, array_serializer
    ]
    """`two_power_zero_edge`: `[center, alpha, beta]`, `center` in keV.
    `spline_knots`: `log(Te)` of the spline component at each knot."""

    rho_knots: (
        Annotated[jt.Float[NpOrJaxArray, " n_knot"], array_validator, array_serializer]
        | None
    ) = None
    """Radial grid of `spline_knots`, shared by both profiles; None for the others."""

    @pydantic.model_validator(mode="after")
    def _check_params_match_the_family(self) -> Self:
        density_size = self.electron_density_params.shape[0]
        temperature_size = self.electron_temperature_params.shape[0]
        if self.family == "two_power_zero_edge":
            n_expected = len(TWO_POWER_ZERO_EDGE_PARAM_NAMES)
            for size, name in (
                (density_size, "electron_density_params"),
                (temperature_size, "electron_temperature_params"),
            ):
                if size != n_expected:
                    raise ValueError(
                        f"{name} has {size} entries but two_power_zero_edge takes "
                        f"{n_expected}: {TWO_POWER_ZERO_EDGE_PARAM_NAMES}."
                    )
        elif self.family == "spline_knots":
            if self.rho_knots is None:
                raise ValueError("spline_knots needs rho_knots.")
            for size, name in (
                (density_size, "electron_density_params"),
                (temperature_size, "electron_temperature_params"),
            ):
                if size != self.rho_knots.shape[0]:
                    raise ValueError(
                        f"{name} has {size} entries but rho_knots has "
                        f"{self.rho_knots.shape[0]}; spline_knots takes one "
                        f"log-value per knot."
                    )
        return self


DENSITY_UNIT = 1.0e19
"""m^-3. The `two_power_zero_edge` density `center` is expressed in units of
`1e19 m^-3` (the convention of Merlo et al. 2023, JINST 18 P11012), so the sampled and
optimized value is O(1) instead of O(1e19); `electron_density` converts back to SI."""

TEMPERATURE_UNIT = 1.0e3
"""eV. The `two_power_zero_edge` temperature `center` is expressed in keV, for the same
reason."""


def evaluate_profile(
    family: ProfileFamily,
    params: jt.Float[NpOrJaxArray, " n_param"],
    rho: jt.Float[NpOrJaxArray, " n_rho"],
    rho_knots: jt.Float[NpOrJaxArray, " n_knot"] | None = None,
) -> jt.Float[jax.Array, " n_rho"]:
    """Evaluate one scalar profile of the given family on a radial grid.

    The result is in the family's own units, which for `two_power_zero_edge` are the
    scaled ones; `to_si` converts.

    Args:
        family: Which parametrization `params` is expressed in.
        params: The parameter vector of one profile.
        rho: Radial coordinates to evaluate at.
        rho_knots: Knot locations for `spline_knots`, None otherwise.

    Returns:
        The profile values at `rho`.
    """
    if family == "two_power_zero_edge":
        return _two_power_zero_edge(jnp.asarray(params), jnp.asarray(rho))
    if family == "spline_knots":
        assert rho_knots is not None, "spline_knots requires rho_knots"
        return _spline_knots(
            jnp.asarray(params), jnp.asarray(rho), jnp.asarray(rho_knots)
        )
    raise ValueError(f"unknown profile family: {family}")


def electron_density(
    profiles: PlasmaProfiles,
    rho: jt.Float[NpOrJaxArray, " n_rho"],
) -> jt.Float[jax.Array, " n_rho"]:
    """Electron density at `rho`.

    Args:
        profiles: The parametrized `ne`/`Te` pair.
        rho: Radial coordinates to evaluate at.

    Returns:
        The electron density in m^-3.
    """
    return _profile_si(profiles, profiles.electron_density_params, rho, DENSITY_UNIT)


def electron_temperature(
    profiles: PlasmaProfiles,
    rho: jt.Float[NpOrJaxArray, " n_rho"],
) -> jt.Float[jax.Array, " n_rho"]:
    """Electron temperature at `rho`.

    Args:
        profiles: The parametrized `ne`/`Te` pair.
        rho: Radial coordinates to evaluate at.

    Returns:
        The electron temperature in eV.
    """
    return _profile_si(
        profiles, profiles.electron_temperature_params, rho, TEMPERATURE_UNIT
    )


def to_si(
    family: ProfileFamily,
    value: jt.Float[NpOrJaxArray, " n_rho"],
    unit: float,
) -> jt.Float[jax.Array, " n_rho"]:
    """Convert a profile value from its family's own units to SI.

    The log-space families are already SI after the exponential; the others carry
    scaled parameters and are rescaled. Public because a raw sample band needs it too.

    Args:
        family: Which parametrization `value` was evaluated from.
        value: Profile values in the family's own units.
        unit: SI value of one scaled unit, i.e. `DENSITY_UNIT` or `TEMPERATURE_UNIT`.

    Returns:
        The same profile in SI.
    """
    if family == "spline_knots":
        return jnp.asarray(value)
    return jnp.asarray(value) * unit


def from_si(
    family: ProfileFamily,
    value: jt.Float[NpOrJaxArray, " n_rho"],
    unit: float,
) -> jt.Float[jax.Array, " n_rho"]:
    """Convert a profile value from SI into its family's own units, inverting `to_si`.

    Its own function because getting it wrong is silent: dividing a `spline_knots`
    target by `DENSITY_UNIT` puts its log values 43.7 too small, reading as a bad fit.

    Args:
        family: Which parametrization the value is being expressed in.
        value: Profile values in SI.
        unit: SI value of one scaled unit, i.e. `DENSITY_UNIT` or `TEMPERATURE_UNIT`.

    Returns:
        The same profile in the family's own units.
    """
    if family == "spline_knots":
        return jnp.asarray(value)
    return jnp.asarray(value) / unit


def pressure_pascal(
    profiles: PlasmaProfiles,
    rho: jt.Float[NpOrJaxArray, " n_rho"],
) -> jt.Float[jax.Array, " n_rho"]:
    """Total thermal pressure `p = 2 * ne * Te` at `rho`, the surrogate's input.

    Args:
        profiles: The parametrized `ne`/`Te` pair.
        rho: Radial coordinates to evaluate at, normally the surrogate's own grid.

    Returns:
        The pressure in Pa.
    """
    return (
        2.0
        * thomson_scattering_utils.ELEMENTARY_CHARGE
        * electron_density(profiles, rho)
        * electron_temperature(profiles, rho)
    )


def _two_power_zero_edge(
    params: jt.Float[jax.Array, " 3"],
    rho: jt.Float[jax.Array, " n_rho"],
) -> jt.Float[jax.Array, " n_rho"]:
    """`center * (1 - rho^alpha)^beta`, exactly zero at `rho = 1`; the family of Merlo
    et al. (2023, JINST 18 P11012) restricted to a vanishing edge."""
    center, alpha, beta = params[0], params[1], params[2]
    # Clip the fractional-power base so rho >= 1, i.e. extrapolation beyond the
    # boundary, stays finite instead of going negative under the outer power.
    base = jnp.clip(1.0 - _safe_pow(rho, alpha), 0.0, 1.0)
    return center * _safe_pow(base, beta)


def _safe_pow(
    base: jt.Float[jax.Array, " n_rho"],
    exponent: jt.Float[jax.Array, ""],
) -> jt.Float[jax.Array, " n_rho"]:
    """`base ** exponent` with a finite gradient where `base == 0`.

    `d(x ** p) / dp = x ** p log(x)` is nan at `x == 0`, which the base hits on axis and
    outside the boundary, so the power is taken on a dummy `1.0` there and masked back.
    """
    safe_base = jnp.where(base > 0.0, base, 1.0)
    return jnp.where(base > 0.0, safe_base**exponent, 0.0)


def _spline_knots(
    log_values: jt.Float[jax.Array, " n_knot"],
    rho: jt.Float[jax.Array, " n_rho"],
    rho_knots: jt.Float[jax.Array, " n_knot"],
) -> jt.Float[jax.Array, " n_rho"]:
    """`exp(interp(rho^2, rho_knots^2, log_values)) * (1 - rho^2)`.

    Interpolating in `s = rho^2` makes the profile flat on axis and the envelope makes
    it exactly zero at the boundary; the clip holds extrapolation past it at zero.
    """
    s = jnp.square(rho)
    spline = jnp.exp(jnp.interp(s, jnp.square(rho_knots), log_values))
    return spline * jnp.clip(1.0 - s, 0.0, None)


def _profile_si(
    profiles: PlasmaProfiles,
    params: jt.Float[NpOrJaxArray, " n_param"],
    rho: jt.Float[NpOrJaxArray, " n_rho"],
    unit: float,
) -> jt.Float[jax.Array, " n_rho"]:
    """Evaluate one of the pair's profiles at `rho` and convert it to SI."""
    value = evaluate_profile(profiles.family, params, rho, profiles.rho_knots)
    return to_si(profiles.family, value, unit)
