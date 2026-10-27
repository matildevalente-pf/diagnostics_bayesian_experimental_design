# The one NumPyro model every solver in this folder targets: it samples the `ne`/`Te`
# parameters, and one Thomson calibration correction per volume if asked, and scores
# them on the data.

import functools
import zlib
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import numpyro
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_target_types,
    inference_types,
    likelihood_utils,
    priors_utils,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    model_storage_utils,
    surrogate_utils,
)

_START_CENTERS: dict[str, dict[str, float]] = {
    "two_power_zero_edge": {"ne_0": 9.5, "te_0": 5.1},
}
"""Axis values of `initial_values`, per family, below the catalogue truths in pressure
at `diagnostic_design_point.MACHINE_SCALE`; every calibration correction sits at one
and every other site at its prior mean.

In the family's scaled units.
"""

_START_SEED_OFFSET = 4
"""Offset of the MAP starts' key from the noise seed; the noise, the chains and the
Laplace draws take offsets zero to three."""


def precompute_flux_guess(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> jt.Float[jax.Array, "n_point 2"] | None:
    """A one-time `(rho, theta)` seed for every diagnostic point, computed eagerly.

    Pays the solve's 384-node grid search once instead of per objective evaluation,
    which the fixed boundary makes sound. The seed must not depend on the parameters.

    Args:
        settings: The run configuration.
        priors: Priors on both profiles, whose start point fixes the seed equilibrium.
        target: The observed data, which fixes the points.
        surrogate: The equilibrium forward model.
        fixed_coefficients: Geometry to seed against instead of the surrogate's
            prediction at the prior median; None calls the surrogate.

    Returns:
        The per-point seed, or None when the target carries no diagnostics.
    """
    points = likelihood_utils.diagnostic_points(
        target, settings.interferometer_n_sample
    )
    if points is None:
        return None
    if fixed_coefficients is not None:
        rho, theta = desc_utils.initial_guess(fixed_coefficients, points)
        return jnp.stack([rho, theta], axis=1)
    profiles = profiles_from_point(
        settings, initial_values(priors, settings.profile_family)
    )
    rho, theta = desc_utils.initial_guess(
        equilibrium_coefficients(surrogate, profiles, None), points
    )
    return jnp.stack([rho, theta], axis=1)


def calibration_corrections(
    point: Mapping[str, jax.typing.ArrayLike],
    thomson_target: inference_target_types.ThomsonSignalTarget | None,
) -> jt.Float[jax.Array, " n_volume"] | None:
    """The per-volume calibration corrections a point carries, None when it has none.

    Args:
        point: One value per sample site.
        thomson_target: The Thomson data, whose volume count fixes the vector's length.

    Returns:
        `delta_j` per volume, exactly one on the volumes with no inferred correction.
    """
    if thomson_target is None:
        return None
    names = priors_utils.site_names(point, priors_utils.CALIBRATION_PREFIX)
    if not names:
        return None
    n_volume = np.shape(thomson_target.global_calibration)[0]
    indices = jnp.asarray([int(name.split("_")[1]) for name in names])
    values = jnp.stack([jnp.asarray(point[name]) for name in names])
    return jnp.ones(n_volume).at[indices].set(values)


def initial_values(
    priors: priors_utils.ProfilePriors,
    profile_family: profile_parametrization_types.ProfileFamily,
) -> dict[str, jt.Float[jax.Array, ""]]:
    """Per-site starting values in the constrained space: the family's axis values of
    `_START_CENTERS`, every calibration correction at one, every other site at its
    prior mean.

    Args:
        priors: Priors on both profiles.
        profile_family: Which family's start table to read.

    Returns:
        One value per sample site, keyed `ne_<index>` and `te_<index>`.
    """
    if profile_family not in _START_CENTERS:
        raise ValueError(
            f"no start point for profile family {profile_family!r}; only the families "
            f"`priors_utils.default_priors` builds have one."
        )
    values = {name: jnp.asarray(prior.mean) for name, prior in priors.items()}
    values.update(
        {
            name: jnp.asarray(value)
            for name, value in _START_CENTERS[profile_family].items()
        }
    )
    # The centre of the log-uniform box, which its mean `(N - 1/N) / (2 ln N)` is not.
    values.update(
        {
            name: jnp.asarray(1.0)
            for name in priors_utils.site_names(priors, priors_utils.CALIBRATION_PREFIX)
        }
    )
    return values


def start_points(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
) -> list[dict[str, jt.Float[jax.Array, ""]]]:
    """Where each MAP run starts: one draw from every prior per start, keyed by the
    noise seed, the start and the site, so a single start is the first of any count.

    Args:
        settings: The run configuration, supplying the family, the seed and the count.
        priors: Priors on both profiles.

    Returns:
        One start per draw a run may use, `n_starts` or up to `max_attempts`, each one
        value per sample site in the constrained space.
    """
    key = jax.random.PRNGKey(settings.seed + _START_SEED_OFFSET)
    # Folding in the site name, not its position, gives every arm of a truth the same
    # profile starts whatever calibration sites it adds.
    return [
        {
            name: jnp.asarray(
                prior.sample(
                    jax.random.fold_in(
                        jax.random.fold_in(key, start), zlib.crc32(name.encode())
                    )
                )
            )
            for name, prior in priors.items()
        }
        for start in range(max(settings.map.n_starts, settings.map.max_attempts))
    ]


def profiles_from_point(
    settings: inference_types.PlasmaEstimationSettings,
    point: Mapping[str, jt.Float[NpOrJaxArray, ""]],
) -> profile_parametrization_types.PlasmaProfiles:
    """The profile pair one per-site point describes, on this run's family and grid.

    Args:
        settings: The run configuration, supplying the family and its hyperparameters.
        point: One value per sample site, keyed `ne_<index>` and `te_<index>`.

    Returns:
        The profiles those parameters describe.
    """
    return profile_parametrization_types.PlasmaProfiles(
        family=settings.profile_family,
        electron_density_params=jnp.stack(
            [point[name] for name in priors_utils.site_names(point, "ne")]
        ),
        electron_temperature_params=jnp.stack(
            [point[name] for name in priors_utils.site_names(point, "te")]
        ),
    )


def plasma_model(
    *,
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    flux_guess: jt.Float[NpOrJaxArray, "n_point 2"] | None,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> None:
    """Sample the profile parameters from their priors and score them on the data.

    Keyword-only, so a caller binds the configuration with `functools.partial` into the
    no-argument callable NumPyro traces. `fixed_coefficients` freezes the geometry.

    Args:
        settings: The run configuration.
        priors: Priors on both profiles.
        target: The observed data.
        surrogate: The equilibrium forward model.
        flux_guess: Precomputed per-point flux seed, or None.
        fixed_coefficients: Geometry to hold frozen for every parameter value; None
            predicts it from the pressure with the surrogate. Pass the matching
            `precompute_flux_guess` seed with it, or the seed describes a different
            equilibrium from the one being inverted against.
    """
    sampled = {name: numpyro.sample(name, prior) for name, prior in priors.items()}
    electron_density_params = jnp.stack(
        [sampled[name] for name in priors_utils.site_names(priors, "ne")]
    )
    electron_temperature_params = jnp.stack(
        [sampled[name] for name in priors_utils.site_names(priors, "te")]
    )
    profiles = profile_parametrization_types.PlasmaProfiles(
        family=settings.profile_family,
        electron_density_params=electron_density_params,
        electron_temperature_params=electron_temperature_params,
    )
    numpyro.factor(
        "log_likelihood",
        likelihood_utils.log_likelihood(
            profiles,
            equilibrium_coefficients(surrogate, profiles, fixed_coefficients),
            target,
            maxiter=settings.inverse_mapping_maxiter,
            interferometer_n_sample=settings.interferometer_n_sample,
            flux_guess=flux_guess,
            calibration_corrections=calibration_corrections(
                sampled, target.thomson_signals
            ),
        ),
    )


def bind_plasma_model(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> Callable[[], None]:
    """Bind `plasma_model` to a run, precomputing its knots and flux seed once.

    Args:
        settings: The run configuration.
        priors: Priors on both profiles.
        target: The observed data.
        surrogate: The equilibrium forward model.
        fixed_coefficients: Geometry to hold frozen; None solves for it.

    Returns:
        The no-argument callable NumPyro traces.
    """
    return functools.partial(
        plasma_model,
        settings=settings,
        priors=priors,
        target=target,
        surrogate=surrogate,
        flux_guess=precompute_flux_guess(
            settings, priors, target, surrogate, fixed_coefficients
        ),
        fixed_coefficients=fixed_coefficients,
    )


def equilibrium_coefficients(
    surrogate: model_storage_utils.LoadedSurrogate,
    profiles: profile_parametrization_types.PlasmaProfiles,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None,
) -> desc_types.DescFourierZernikeCoefficients:
    """The geometry one profile pair implies, or the frozen one it was handed.

    The one place the surrogate is called: it is float32 while the flux solve needs
    float64 for Zernike conditioning, so the cast happens here and nowhere else.

    Args:
        surrogate: The equilibrium forward model.
        profiles: The profile pair whose pressure the surrogate is called on.
        fixed_coefficients: Geometry to return unchanged, skipping the surrogate; None
            predicts it.

    Returns:
        The Fourier-Zernike coefficients to invert the diagnostics against.
    """
    if fixed_coefficients is not None:
        return fixed_coefficients
    return surrogate_utils.unpack_flat_target(
        surrogate.basis,
        jnp.asarray(
            surrogate.predict(
                profile_parametrization_types.pressure_pascal(
                    profiles, surrogate.basis.pressure_rho_grid
                )
            ),
            dtype=jnp.float64,
        ),
    ).coefficients
