# The Gaussian data likelihood: the whitened residual vector and its summed
# log-density. `residuals` is the primitive, `log_likelihood` a sum over it.

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_target_types,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    interferometer_utils,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.inverse_mapping import (  # noqa: E501
    inverse_map_utils,
)

_THOMSON_KIND = "thomson"
_CHORD_KIND = "chord"


class LikelihoodResiduals(NamedTuple):
    """The whitened data residual of one parameter set, entry per compared channel.

    The per-channel fields share one static length, `n_volume * n_channel` Thomson
    entries followed by `n_chord` interferometer entries.
    """

    residual: jt.Float[jax.Array, " n_residual"]
    """`(predicted - observed) / sigma`, exactly zero where `valid` is False."""

    valid: jt.Bool[jax.Array, " n_residual"]
    """Whether the channel carries likelihood weight.

    A mask and not a boolean index, because the residual keeps one static length; a
    masked row is then exactly zero in `jacrev` and drops out of `J^T J`.
    """

    sigma: jt.Float[jax.Array, " n_residual"]
    """The 1-sigma uncertainty each residual was whitened by.

    One on masked entries, so `log(sigma)` stays finite.
    """


class PointGroup(NamedTuple):
    """One diagnostic's real-space query points, in the likelihood's fixed order."""

    kind: str
    """`thomson` for the scattering volumes, `chord` for one interferometer chord."""

    points: jt.Float[jax.Array, "n_point 3"]
    """Cartesian points this group needs inverted to flux coordinates."""


def point_groups(
    target: inference_target_types.InferenceTarget,
    interferometer_n_sample: int = 64,
) -> list[PointGroup]:
    """Every point the likelihood inverts, grouped by diagnostic and in its own order.

    Thomson volumes first, then one group per interferometer chord. This ordering is the
    contract between `diagnostic_points`, `residuals` and any precomputed flux seed.

    Args:
        target: The observed data, which fixes what is present.
        interferometer_n_sample: Points per chord in the line integral.

    Returns:
        The groups, empty when the target carries no diagnostics.
    """
    groups: list[PointGroup] = []
    if target.thomson_signals is not None:
        groups.append(
            PointGroup(
                kind=_THOMSON_KIND,
                points=jnp.asarray(target.thomson_signals.geometry.volume_positions),
            )
        )
    if target.interferometer is not None:
        for chord in target.interferometer.geometries:
            groups.append(
                PointGroup(
                    kind=_CHORD_KIND,
                    points=interferometer_utils.chord_sample_points(
                        chord, interferometer_n_sample
                    ),
                )
            )
    return groups


def diagnostic_points(
    target: inference_target_types.InferenceTarget,
    interferometer_n_sample: int = 64,
) -> jt.Float[jax.Array, "n_point 3"] | None:
    """All real-space points the likelihood inverts, concatenated in its own order.

    Use it to precompute a `(rho, theta)` seed once per run and pass it back as
    `flux_guess`; the points are fixed data, so the seed never needs recomputing.

    Args:
        target: The observed data.
        interferometer_n_sample: Points per chord in the line integral.

    Returns:
        The points, or None when the target carries no diagnostics.
    """
    groups = point_groups(target, interferometer_n_sample)
    if not groups:
        return None
    return jnp.concatenate([group.points for group in groups], axis=0)


def residuals(
    profiles: profile_parametrization_types.PlasmaProfiles,
    coefficients: desc_types.DescFourierZernikeCoefficients,
    target: inference_target_types.InferenceTarget,
    tol: float = 1e-12,
    maxiter: int = 50,
    interferometer_n_sample: int = 64,
    flux_guess: jt.Float[NpOrJaxArray, "n_point 2"] | None = None,
    calibration_corrections: jt.Float[NpOrJaxArray, " n_volume"] | None = None,
) -> LikelihoodResiduals:
    """Whitened residual of every compared channel, for one parameter set.

    Every point needing the flux inversion is solved in one batched call, differentiable
    in forward and reverse mode through the implicit rule of `implicit_flux_solve`.

    Args:
        profiles: The parametrized `ne`/`Te` pair being tested.
        coefficients: Equilibrium geometry the points are inverted against.
        target: The observed data.
        tol: Real-space convergence tolerance of the Newton solve, in m.
        maxiter: Number of Newton iterations, a static Python int.
        interferometer_n_sample: Points per chord in the line integral.
        flux_guess: Optional `(rho, theta)` seed for exactly the points
            `diagnostic_points` returns, in the same order.
        calibration_corrections: Inferred `delta_j` multiplying the target's nominal
            calibration volume by volume; None takes the nominal one as known.

    Returns:
        The residual, its validity mask and the uncertainties it was divided by.
    """
    groups = point_groups(target, interferometer_n_sample)
    if not groups:
        empty = jnp.zeros((0,))
        return LikelihoodResiduals(
            residual=empty,
            valid=jnp.zeros((0,), dtype=bool),
            sigma=empty,
        )

    all_points = jnp.concatenate([group.points for group in groups], axis=0)
    if flux_guess is not None:
        assert flux_guess.shape[0] == all_points.shape[0], (
            f"flux_guess has {flux_guess.shape[0]} rows but the target has "
            f"{all_points.shape[0]} diagnostic points; build it with "
            f"`diagnostic_points(target, interferometer_n_sample)`."
        )
    flux_rho, _ = inverse_map_utils.implicit_flux_solve(
        coefficients, all_points, guess=flux_guess, tol=tol, maxiter=maxiter
    )
    electron_density = profile_parametrization_types.electron_density(
        profiles, flux_rho
    )
    electron_temperature = profile_parametrization_types.electron_temperature(
        profiles, flux_rho
    )

    parts: list[LikelihoodResiduals] = []
    chord_line_integrals: list[jax.Array] = []
    offset = 0
    for group in groups:
        size = group.points.shape[0]
        group_slice = slice(offset, offset + size)
        offset += size
        if group.kind == _THOMSON_KIND:
            signals_target = target.thomson_signals
            assert signals_target is not None, "a thomson group needs a thomson target"
            predicted = thomson_scattering_utils.thomson_signals(
                signals_target.geometry,
                signals_target.response,
                electron_density[group_slice],
                electron_temperature[group_slice],
                thomson_scattering_utils.effective_calibration(
                    signals_target.global_calibration, calibration_corrections
                ),
            )
            parts.append(
                _masked_residual(
                    jnp.ravel(predicted),
                    jnp.ravel(jnp.asarray(signals_target.signals)),
                    jnp.ravel(jnp.asarray(signals_target.uncertainties)),
                )
            )
        else:
            # Zero the density outside the last closed flux surface: the chord crosses
            # the plasma edge, and there is no core density to integrate beyond it.
            core_density = jnp.where(
                flux_rho[group_slice] <= 1.0, electron_density[group_slice], 0.0
            )
            chord_line_integrals.append(
                interferometer_utils.line_integrated_density_signal(
                    core_density, group.points
                )
            )

    if target.interferometer is not None:
        parts.append(
            _masked_residual(
                jnp.stack(chord_line_integrals),
                jnp.asarray(target.interferometer.line_integrated_density),
                jnp.asarray(target.interferometer.line_integrated_density_sigma),
            )
        )
    return LikelihoodResiduals(
        residual=jnp.concatenate([part.residual for part in parts]),
        valid=jnp.concatenate([part.valid for part in parts]),
        sigma=jnp.concatenate([part.sigma for part in parts]),
    )


def log_likelihood(
    profiles: profile_parametrization_types.PlasmaProfiles,
    coefficients: desc_types.DescFourierZernikeCoefficients,
    target: inference_target_types.InferenceTarget,
    tol: float = 1e-12,
    maxiter: int = 50,
    interferometer_n_sample: int = 64,
    flux_guess: jt.Float[NpOrJaxArray, "n_point 2"] | None = None,
    calibration_corrections: jt.Float[NpOrJaxArray, " n_volume"] | None = None,
) -> jt.Float[jax.Array, ""]:
    """Independent-Gaussian log-likelihood of every diagnostic present in `target`.

    Args:
        profiles: The parametrized `ne`/`Te` pair being tested.
        coefficients: Equilibrium geometry the points are inverted against.
        target: The observed data.
        tol: Real-space convergence tolerance of the Newton solve, in m.
        maxiter: Number of Newton iterations, a static Python int.
        interferometer_n_sample: Points per chord in the line integral.
        flux_guess: Optional `(rho, theta)` seed, as for `residuals`.
        calibration_corrections: Inferred `delta_j` per volume, as for `residuals`.

    Returns:
        The summed log-density over the valid channels; zero for an empty target.
    """
    whitened = residuals(
        profiles,
        coefficients,
        target,
        tol,
        maxiter,
        interferometer_n_sample=interferometer_n_sample,
        flux_guess=flux_guess,
        calibration_corrections=calibration_corrections,
    )
    per_channel = -0.5 * (
        jnp.square(whitened.residual)
        + math.log(2.0 * math.pi)
        + 2.0 * jnp.log(whitened.sigma)
    )
    return jnp.sum(jnp.where(whitened.valid, per_channel, 0.0))


def _masked_residual(
    predicted: jt.Float[jax.Array, " n"],
    observed: jt.Float[jax.Array, " n"],
    sigma: jt.Float[jax.Array, " n"],
) -> LikelihoodResiduals:
    """Whiten one diagnostic's channels, zeroing the ones that carry no information.

    Invalid means a non-finite `observed` or `sigma`, or a `sigma` at or below zero,
    which is how the faint-channel mask and a chord missing the plasma are recorded.
    """
    valid = jnp.isfinite(observed) & jnp.isfinite(sigma) & (sigma > 0.0)
    safe_sigma = jnp.where(valid, sigma, 1.0)
    return LikelihoodResiduals(
        residual=jnp.where(valid, (predicted - observed) / safe_sigma, 0.0),
        valid=valid,
        sigma=safe_sigma,
    )
