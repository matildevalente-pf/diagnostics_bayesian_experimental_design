# What every placement study shares: the truths, the model a layout is linearized in,
# the information gain of a layout and a catalogue of parallel chords in one plane.

import math

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    flat_posterior_utils,
    inference_model_utils,
    inference_target_types,
    inference_types,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    geometry_utils,
    interferometer_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    model_storage_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E501
    parametrization_fit_utils,
)

TRUTH_NAMES = tuple(desc_truth_types.DESC_TRUTH_IDS)
"""Ground truths a layout is scored on, the whole catalogue."""


def flat_posterior(
    settings: inference_types.PlasmaEstimationSettings,
    surrogate: model_storage_utils.LoadedSurrogate,
    thomson_target: inference_target_types.ThomsonSignalTarget | None = None,
) -> tuple[flat_posterior_utils.FlatPosterior, priors_utils.ProfilePriors]:
    """The flattened model a layout is linearized in, with the priors it is built on.

    Args:
        settings: The inference configuration.
        surrogate: The equilibrium forward model.
        thomson_target: Thomson data whose active volumes get a calibration site; None
            leaves the profiles' parameters alone.

    Returns:
        The posterior and its priors.
    """
    priors = inference_model_utils.priors_for(settings, thomson_target)
    posterior = flat_posterior_utils.build_flat_posterior(
        inference_model_utils.bind_plasma_model(
            settings, priors, inference_target_types.InferenceTarget(), surrogate
        ),
        inference_model_utils.initial_values(priors, settings.profile_family),
    )
    return posterior, priors


def truth_point(
    truth: desc_truth_types.DescTruth, posterior: flat_posterior_utils.FlatPosterior
) -> jt.Float[jax.Array, " n_param"]:
    """The truth's two-power fit in the unconstrained coordinates of the inference.

    Args:
        truth: The ground truth.
        posterior: The flattened model whose transforms and site order define `z`.

    Returns:
        The point every layout is linearized at for this truth.

    Raises:
        ValueError: If the fit lies outside the prior box, where `z` does not exist.
    """
    density, temperature = parametrization_fit_utils.target_parameters(
        "two_power_zero_edge", truth.profiles
    )
    values = {f"ne_{index}": value for index, value in enumerate(density)}
    values |= {f"te_{index}": value for index, value in enumerate(temperature)}
    # Synthetic data carry the nominal calibration, so every correction is one.
    values |= {name: 1.0 for name in posterior.site_names if name not in values}
    z = jnp.stack(
        [
            posterior.transforms[name].inv(jnp.asarray(values[name]))
            for name in posterior.site_names
        ]
    )
    if not bool(jnp.all(jnp.isfinite(z))):
        raise ValueError(f"the two-power fit of {truth.name} is outside the prior box")
    return z


def parameter_indices(
    posterior: flat_posterior_utils.FlatPosterior, prefixes: tuple[str, ...]
) -> jt.Int[np.ndarray, " n_interest"]:
    """Positions in `z` of the sites whose name starts with one of `prefixes`.

    Args:
        posterior: The flattened model whose site order defines `z`.
        prefixes: Site name prefixes, such as `("ne_",)` or `("ne_", "te_")`.

    Returns:
        The indices, in `z` order.
    """
    return np.array(
        [
            index
            for index, name in enumerate(posterior.site_names)
            if name.startswith(prefixes)
        ]
    )


def information_gain_bits(
    information: jt.Float[jax.Array, "*batch n_param n_param"],
    precision: jt.Float[jax.Array, "*batch n_param n_param"],
    interest: jt.Int[np.ndarray, " n_interest"],
) -> jt.Float[jax.Array, "*batch"]:
    """Expected Kullback-Leibler divergence from prior to posterior of the parameters of
    interest, in bits, under the Gaussian approximation; the rest are nuisances.

    Args:
        information: Gauss-Newton data information `J^T J` of each layout.
        precision: Prior precision at the linearization point, broadcast against
            `information`.
        interest: Indices of the parameters of interest.

    Returns:
        `0.5 log2(det C_prior / det C_posterior)` of the interest block, per layout.
    """
    block = (..., interest[:, None], interest[None, :])
    prior = jnp.linalg.inv(precision)[block]
    posterior = jnp.linalg.inv(information + precision)[block]
    gain = jnp.linalg.slogdet(prior)[1] - jnp.linalg.slogdet(posterior)[1]
    return 0.5 * gain / math.log(2.0)


def parallel_chords(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: float,
    direction_angle: float,
    n_chords: int,
    margin: float = 0.025,
) -> tuple[
    jt.Float[np.ndarray, " n_chord"],
    jt.Float[np.ndarray, "n_chord 2"],
    jt.Float[jax.Array, "n_chord 3"],
    jt.Float[jax.Array, "n_chord 3"],
]:
    """`n_chords` parallel chords spread evenly across the plasma of one poloidal plane.

    Args:
        coefficients: Equilibrium whose boundary and design surface the chords cross.
        toroidal_angle: Plane of the chords, in rad.
        direction_angle: Direction of the chords in the `(R, Z)` plane, in rad.
        n_chords: Number of chords.
        margin: Share of the plasma's width left out on each side.

    Returns:
        Each chord's offset from the design-surface centre along the chord normal in m,
        its two surface angles in rad, and the start and end of its in-plasma segment.
    """
    plasma, surface = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, geometry_utils.DEFAULT_SURFACE_OFFSET
    )
    centre = np.asarray(surface.centre)
    tangent = np.array([math.cos(direction_angle), math.sin(direction_angle)])
    normal = np.array([-tangent[1], tangent[0]])
    projection = normal @ (np.asarray(plasma) - centre[:, None])
    width = projection.max() - projection.min()
    offsets = np.linspace(
        projection.min() + margin * width, projection.max() - margin * width, n_chords
    )
    return chords_at_offsets(coefficients, toroidal_angle, direction_angle, offsets)


def chords_at_offsets(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: float,
    direction_angle: float,
    offsets: jt.Float[NpOrJaxArray, " n_chord"],
) -> tuple[
    jt.Float[np.ndarray, " n_chord"],
    jt.Float[np.ndarray, "n_chord 2"],
    jt.Float[jax.Array, "n_chord 3"],
    jt.Float[jax.Array, "n_chord 3"],
]:
    """Parallel chords of one poloidal plane at given offsets from the surface centre.

    Args:
        coefficients: Equilibrium whose boundary and design surface the chords cross.
        toroidal_angle: Plane of the chords, in rad.
        direction_angle: Direction of the chords in the `(R, Z)` plane, in rad.
        offsets: Each chord's offset along the chord normal, in m.

    Returns:
        The offsets, each chord's two surface angles in rad, and the start and end of
        its in-plasma segment.
    """
    _, surface = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, geometry_utils.DEFAULT_SURFACE_OFFSET
    )
    centre = np.asarray(surface.centre)
    tangent = np.array([math.cos(direction_angle), math.sin(direction_angle)])
    normal = np.array([-tangent[1], tangent[0]])
    offsets = np.asarray(offsets, dtype=float)
    surface_curve = np.asarray(surface.curve)
    reach = 2.0 * float(np.max(np.linalg.norm(surface_curve - centre[:, None], axis=0)))
    surface_angles, chords = [], []
    for index, offset in enumerate(offsets):
        start = centre + offset * normal - reach * tangent
        end = centre + offset * normal + reach * tangent
        span = geometry_utils.segment_curve_span(surface_curve, start, end)
        assert span is not None, "a chord through the plasma always meets the surface"
        ends = np.stack([start + fraction * (end - start) for fraction in span], axis=1)
        surface_angles.append(np.arctan2(ends[1] - centre[1], ends[0] - centre[0]))
        lab = np.asarray(geometry_utils.lab_frame_points(ends, toroidal_angle))
        chords.append(
            interferometer_utils.build_geometry_from_endpoints(
                lab[0], lab[1], f"chord_{index}"
            )
        )
    trimmed = geometry_utils.trim_chords_to_plasma(chords, coefficients)
    return (
        offsets,
        np.asarray(surface_angles),
        jnp.stack([jnp.asarray(chord.chord_start) for chord in trimmed]),
        jnp.stack([jnp.asarray(chord.chord_end) for chord in trimmed]),
    )
