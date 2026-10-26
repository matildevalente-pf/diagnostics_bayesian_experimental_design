# Turning a ground truth plus a diagnostic design point into the noisy data an inference
# is fitted against.
#
# There is no experiment to fit. A known true state is pushed through the same
# synthetic Thomson and interferometer forward models the likelihood uses, on a
# prescribed layout, and measurement noise is added; that is the target.
#
# Every noise and calibration property lives on the `DiagnosticDesignPoint`, so a
# stored design fully specifies the synthetic instrument and the truth generation and
# the layout cannot drift apart. The truths are DESC equilibria read by
# `desc_truth_types`.
#
# Each chord's sample points go through the flux solve and the profile evaluators
# directly, exactly as the likelihood does them, so the truth and the fit cannot end
# up sampling differently.
#
# The random draws are gated in a fixed order, so a given seed reproduces its target
# whichever optional noise terms are switched on.

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    inference_target_types,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
    geometry_utils,
    interferometer_types,
    interferometer_utils,
    thomson_scattering_types,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.inverse_mapping import (  # noqa: E501
    inverse_map_utils,
)


def build_diagnostic_placement(
    design: diagnostic_design_point.DiagnosticDesignPoint,
    reference: desc_truth_types.DescTruth,
    verbose: bool = True,
) -> diagnostic_design_point.DiagnosticPlacement:
    """Build the instrument once, on the reference equilibrium, for every truth.

    The volumes and chords are placed on the reference geometry and, unless
    `design.band_scale_rule` is `none`, each polychromator's bands are scaled for the
    state the reference predicts at its volume (`apply_band_scale_rule`).

    Args:
        design: The design point to build.
        reference: The equilibrium a reconstruction may assume before the profiles are
            known, `desc_truth_types.REFERENCE_EQUILIBRIUM_ID`.
        verbose: Whether to print the detector calibration and the band scales.

    Returns:
        The placement, identified by the content hash `placement_id`.
    """
    geometry, response, chords = diagnostic_design_point.build_diagnostic_geometry(
        reference.coefficients, design
    )
    # `kappa` is read off the pinned reference configuration with the published W7-X
    # bands, never off `design`: the detector is the same whatever layout collects the
    # light for it, and the band rule below needs it before it can weight a channel.
    reference_design = diagnostic_design_point.DiagnosticDesignPoint()
    reference_geometry, reference_response, _ = (
        diagnostic_design_point.build_diagnostic_geometry(
            reference.coefficients, reference_design
        )
    )
    assert reference_geometry is not None, "the pinned design point carries volumes"
    signal_scale = thomson_scattering_utils.electronic_signal_scale(
        reference_geometry,
        reference_response,
        thomson_scattering_utils.absolute_calibration(reference_geometry),
    )
    if verbose:
        print(
            f"[thomson] kappa = {signal_scale:.3e} V s per photoelectron, from the "
            f"reference configuration at ne_min = "
            f"{thomson_scattering_utils.MINIMUM_DETECTABLE_DENSITY:.1e} m^-3, te_min = "
            f"{thomson_scattering_utils.MINIMUM_DETECTABLE_TEMPERATURE_EV:g} eV"
        )
    if design.band_scale_rule != "none" and geometry is not None:
        response = apply_band_scale_rule(
            geometry, response, reference, design, signal_scale, verbose
        )
    return diagnostic_design_point.DiagnosticPlacement(
        design=design,
        reference_equilibrium_id=reference.name,
        thomson_geometry=geometry,
        polychromator_response=response,
        thomson_signal_scale=signal_scale,
        interferometer_chords=chords,
    )


def build_synthetic_target(
    truth: desc_truth_types.DescTruth,
    placement: diagnostic_design_point.DiagnosticPlacement,
    interferometer_n_sample: int = 64,
    trim_chords_to_plasma: bool = True,
    min_expected_counts: float = (thomson_scattering_utils.MIN_EXPECTED_COUNTS),
    true_calibration_factors: jt.Float[NpOrJaxArray, " n_volume"] | None = None,
    seed: int = 0,
    add_noise: bool = True,
    interferometer_sigma: float = interferometer_utils.LINE_DENSITY_NOISE_M2,
) -> inference_target_types.InferenceTarget:
    """Forward-model a ground truth on a placement and add measurement noise.

    Thomson noise follows the placement's `thomson_noise_model`, independent per
    channel and drawn about the noiseless signal, and the Thomson signals are generated
    with the nominal geometric calibration unless per-volume corrections are given;
    interferometer noise is one constant `sigma_IF` per chord, whatever it reads.

    Args:
        truth: Ground-truth state; its profiles generate the signals on the placement's
            instrument, and its geometry maps the instrument's points to flux labels.
        placement: The instrument, built once by `build_diagnostic_placement`.
        interferometer_n_sample: Points per chord in the line integral.
        trim_chords_to_plasma: Whether to integrate each chord over its in-plasma
            segment only; the line of sight itself is the placement's either way.
        min_expected_counts: Faint-channel threshold, in photoelectrons.
        true_calibration_factors: The `delta_j` the data are generated with, one per
            scattering volume; None is the nominal geometric calibration everywhere.
        seed: Seed of every noise draw.
        add_noise: Whether to draw the noise; False records the expected signals with
            the noisy target's uncertainties.
        interferometer_sigma: `sigma_IF` of every chord in m^-2.

    Returns:
        The target, carrying the geometry and calibration it was made with.
    """
    chords = (
        geometry_utils.trim_chords_to_plasma(
            placement.interferometer_chords, truth.coefficients
        )
        if trim_chords_to_plasma
        else placement.interferometer_chords
    )
    line_integrated = np.asarray(
        [
            float(_chord_line_integral(truth, chord, interferometer_n_sample))
            for chord in chords
        ]
    )
    line_sigma = np.asarray(
        interferometer_utils.compute_interferometer_uncertainty(
            line_integrated, interferometer_sigma
        )
    )
    # The Thomson channels draw first, then the chords, so a seed reproduces the same
    # target it always did.
    generator = np.random.default_rng(seed) if add_noise else None
    thomson = (
        None
        if placement.thomson_geometry is None
        else _thomson_signal_target(
            truth,
            placement.design,
            placement.thomson_geometry,
            placement.polychromator_response,
            placement.thomson_signal_scale,
            min_expected_counts,
            true_calibration_factors,
            generator,
        )
    )
    if generator is not None:
        line_integrated = interferometer_utils.draw_recorded_line_integrated_densities(
            generator, line_integrated, interferometer_sigma
        )
    return inference_target_types.InferenceTarget(
        truth_name=truth.name,
        thomson_signals=thomson,
        interferometer=inference_target_types.InterferometerTarget(
            geometries=chords,
            line_integrated_density=line_integrated,
            line_integrated_density_sigma=line_sigma,
        ),
    )


def expected_volume_state(
    geometry: thomson_scattering_types.ThomsonScatteringGeometry,
    reference: desc_truth_types.DescTruth,
) -> tuple[jt.Float[jax.Array, " n_volume"], jt.Float[jax.Array, " n_volume"]]:
    """The `ne` and `Te` a reference equilibrium predicts at every scattering volume.

    Args:
        geometry: The layout, fixing where the volumes sit.
        reference: The equilibrium a reconstruction may assume before the profiles are
            known.

    Returns:
        The density in m^-3 and the temperature in eV, one of each per volume.
    """
    volume_rho, _ = inverse_map_utils.implicit_flux_solve(
        reference.coefficients, geometry.volume_positions
    )
    rho = jnp.clip(volume_rho, 0.0, 1.0)
    return (
        jnp.asarray(
            profile_parametrization_types.electron_density(reference.profiles, rho)
        ),
        jnp.asarray(
            profile_parametrization_types.electron_temperature(reference.profiles, rho)
        ),
    )


def apply_band_scale_rule(
    geometry: thomson_scattering_types.ThomsonScatteringGeometry,
    response: thomson_scattering_types.PolychromatorResponse,
    reference: desc_truth_types.DescTruth,
    design: diagnostic_design_point.DiagnosticDesignPoint,
    signal_scale: float,
    verbose: bool = True,
) -> thomson_scattering_types.PolychromatorResponse:
    """Give every volume the band scale `design.band_scale_rule` picks for the state the
    reference equilibrium predicts at its flux label, at the layout's scattering angle.

    Args:
        geometry: The layout, fixing the volumes and their angles.
        response: The unscaled response the design point built.
        reference: The equilibrium a reconstruction may assume before the profiles are
            known, supplying the expected state.
        design: The design point, naming the rule and the noise model it reads.
        signal_scale: `kappa` of the reference configuration, which the uncertainties
            weighting the Fisher information are evaluated with.
        verbose: Whether to print the resulting scale of every volume.

    Returns:
        The response with `band_scales` set, one factor per volume.
    """
    expected_density, expected_temperature = expected_volume_state(geometry, reference)
    angles = thomson_scattering_utils.volume_scattering_angles(geometry)
    scales = jnp.asarray(
        thomson_scattering_utils.fisher_band_scale(
            expected_density,
            expected_temperature,
            geometry,
            response,
            thomson_scattering_utils.absolute_calibration(geometry),
            design.thomson_noise_model,
            signal_scale,
        )
    )
    if verbose:
        low, high = thomson_scattering_utils.band_scale_bounds(response)
        print(
            f"[thomson] {design.band_scale_rule} band scale per volume, expected state "
            f"from reference equilibrium {reference.name} "
            f"(admissible s in [{low:.3f}, {high:.3f}]):"
        )
        for volume, scale in enumerate(scales):
            clipped = " (clipped)" if scale <= low or scale >= high else ""
            print(
                f"  volume {volume:2d}: theta = {np.degrees(angles[volume]):6.1f} deg, "
                f"expected Te = {expected_temperature[volume]:7.0f} eV, "
                f"s = {scale:.3f}{clipped}"
            )
    return response.model_copy(update={"band_scales": scales})


def readings_on_layout(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    geometry: thomson_scattering_types.ThomsonScatteringGeometry,
    response: thomson_scattering_types.PolychromatorResponse,
    design: diagnostic_design_point.DiagnosticDesignPoint,
    signal_scale: float,
    electron_density_fn: Callable[[NpOrJaxArray], NpOrJaxArray],
    electron_temperature_fn: Callable[[NpOrJaxArray], NpOrJaxArray],
) -> tuple[
    jt.Float[np.ndarray, "n_volume n_channel"],
    jt.Float[np.ndarray, "n_volume n_channel"],
]:
    """What a built layout would record from one plasma state, for a figure to draw.

    The signals and the uncertainties are the ones `build_synthetic_target` gives the
    likelihood, so a figure drawn from this cannot show a precision no inference had.
    Volumes outside the boundary get no light.

    Args:
        coefficients: DESC geometry coefficients of the equilibrium.
        geometry: The layout, which fixes the volumes, the optics and the laser.
        response: Polychromator response (laser and passbands).
        design: The instrument, supplying the noise model.
        signal_scale: `kappa` of the reference configuration, from the placement.
        electron_density_fn: `ne(rho)` of the state to record.
        electron_temperature_fn: `Te(rho)` of the state to record.

    Returns:
        The per-channel signals and their uncertainties, in photoelectrons.
    """
    toroidal_angle = float(geometry.reference_phi)
    positions = np.asarray(geometry.volume_positions)
    volumes_rz = np.stack([np.hypot(positions[:, 0], positions[:, 1]), positions[:, 2]])
    plasma, _ = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, design.surface_offset
    )
    flux_rho = geometry_utils.flux_interpolator(coefficients, toroidal_angle)(
        volumes_rz[0], volumes_rz[1]
    )
    inside = geometry_utils.points_inside(plasma, volumes_rz)
    flux_rho = np.where(inside & np.isfinite(flux_rho), np.nan_to_num(flux_rho), 1.0)
    electron_density = np.where(
        inside, np.asarray(electron_density_fn(jnp.asarray(flux_rho))), 0.0
    )
    # Floored at the minimum detectable temperature: the spectral density needs a
    # positive `Te`, and outside the plasma that state is what a channel would see.
    electron_temperature = np.maximum(
        np.asarray(electron_temperature_fn(jnp.asarray(flux_rho))),
        thomson_scattering_utils.MINIMUM_DETECTABLE_TEMPERATURE_EV,
    )
    calibration = thomson_scattering_utils.absolute_calibration(geometry)
    signals = thomson_scattering_utils.thomson_signals(
        geometry, response, electron_density, electron_temperature, calibration
    )
    return np.asarray(signals), np.asarray(
        thomson_scattering_utils.signal_uncertainties(
            signals, design.thomson_noise_model, signal_scale
        )
    )


def select_chords(
    target: inference_target_types.InferenceTarget,
    chord_indices: tuple[int, ...] | None = None,
) -> inference_target_types.InferenceTarget:
    """Keep only the named interferometer chords, by masking every other one out.

    Masking rather than rebuilding keeps the geometry and the noise draw exactly as they
    were, so a chord kept by two configurations contributes the *same* recorded number
    to both, and the only difference between two cells is how much of that one
    measurement set the likelihood is allowed to see.

    A masked chord is excluded exactly, not merely down-weighted: `likelihood_utils`
    treats a non-positive sigma as invalid, which zeroes its residual and leaves an
    identically zero row in `jacrev(residuals)`, so it drops out of `J^T J` as well. The
    arrays keep their shape, which is what the static-shape contract of the residual
    vector needs.

    Args:
        target: The full target, as `build_synthetic_target` made it.
        chord_indices: Chords to keep; None keeps every chord.

    Returns:
        A copy whose unselected chords carry zero uncertainty.
    """
    if target.interferometer is None or chord_indices is None:
        return target
    sigma = np.asarray(target.interferometer.line_integrated_density_sigma).copy()
    keep = np.zeros(sigma.shape[0], dtype=bool)
    keep[list(chord_indices)] = True
    sigma[~keep] = 0.0
    return target.model_copy(
        update={
            "interferometer": target.interferometer.model_copy(
                update={"line_integrated_density_sigma": sigma}
            )
        }
    )


def thomson_volume_flux_labels(
    volume_positions: jt.Float[NpOrJaxArray, "n_volume 3"],
    truth: desc_truth_types.DescTruth,
) -> jt.Float[np.ndarray, " n_volume"]:
    """Flux label of each Thomson volume, the axis its coverage is measured along.

    Args:
        volume_positions: Cartesian volume centres, from the Thomson geometry.
        truth: Ground truth whose geometry the positions are inverted against.

    Returns:
        `rho` at every volume; values above one lie outside the plasma.
    """
    flux_rho, _ = inverse_map_utils.implicit_flux_solve(
        truth.coefficients, volume_positions
    )
    return np.asarray(flux_rho)


def thomson_local_estimates(
    target: inference_target_types.InferenceTarget,
    truth: desc_truth_types.DescTruth,
) -> tuple[
    jt.Float[np.ndarray, " n_volume"], thomson_scattering_utils.LocalStateEstimate
]:
    """Every scattering volume's own `ne` and `Te`, and the flux label it sits at.

    The recorded signals of one volume are fitted alone, on the instrument the target
    carries, so the estimate reads that volume's spectrum and nothing else.

    Args:
        target: The observed data, carrying the Thomson signals and their instrument.
        truth: Ground truth whose geometry the volume positions are inverted against.

    Returns:
        `rho` per volume and the local estimate at every one of them.
    """
    signals = target.thomson_signals
    assert signals is not None, "a local Thomson estimate needs Thomson signals"
    return (
        thomson_volume_flux_labels(signals.geometry.volume_positions, truth),
        thomson_scattering_utils.local_state_estimate(
            signals.geometry,
            signals.response,
            signals.signals,
            signals.uncertainties,
            signals.global_calibration,
        ),
    )


def _thomson_signal_target(
    truth: desc_truth_types.DescTruth,
    design: diagnostic_design_point.DiagnosticDesignPoint,
    geometry: thomson_scattering_types.ThomsonScatteringGeometry,
    response: thomson_scattering_types.PolychromatorResponse,
    signal_scale: float,
    min_expected_counts: float,
    true_calibration_factors: jt.Float[NpOrJaxArray, " n_volume"] | None,
    generator: np.random.Generator | None,
) -> inference_target_types.ThomsonSignalTarget:
    """One truth's Thomson channels on one layout, noisy when a generator is given."""
    nominal = np.asarray(thomson_scattering_utils.absolute_calibration(geometry))
    # The data are generated with the geometric calibration unless a correction per
    # volume is prescribed; the target records the nominal for the likelihood and the
    # corrections only for coverage checks.
    calibration = np.asarray(
        thomson_scattering_utils.effective_calibration(
            nominal, true_calibration_factors
        )
    )
    volume_rho, _ = inverse_map_utils.implicit_flux_solve(
        truth.coefficients, geometry.volume_positions
    )
    electron_density = profile_parametrization_types.electron_density(
        truth.profiles, volume_rho
    )
    electron_temperature = profile_parametrization_types.electron_temperature(
        truth.profiles, volume_rho
    )
    thomson_scattering_utils.check_incoherent_scattering(
        geometry, response, electron_density, electron_temperature
    )
    signals = thomson_scattering_utils.thomson_signals(
        geometry, response, electron_density, electron_temperature, calibration
    )
    sigma = thomson_scattering_utils.signal_uncertainties(
        signals,
        design.thomson_noise_model,
        signal_scale,
        min_expected_counts=min_expected_counts,
    )
    recorded = (
        np.asarray(signals)
        if generator is None
        else thomson_scattering_utils.draw_recorded_signals(
            generator, signals, sigma, design.thomson_noise_model
        )
    )
    return inference_target_types.ThomsonSignalTarget(
        geometry=geometry,
        response=response,
        signals=recorded,
        uncertainties=np.asarray(sigma),
        global_calibration=nominal,
        true_calibration_factors=true_calibration_factors,
    )


def _chord_line_integral(
    truth: desc_truth_types.DescTruth,
    chord: interferometer_types.InterferometerGeometry,
    interferometer_n_sample: int,
) -> jt.Float[NpOrJaxArray, ""]:
    """The truth's line-integrated density along one chord, sampled as the fit will."""
    sample_points = interferometer_utils.chord_sample_points(
        chord, interferometer_n_sample
    )
    flux_rho, _ = inverse_map_utils.implicit_flux_solve(
        truth.coefficients, sample_points
    )
    electron_density = profile_parametrization_types.electron_density(
        truth.profiles, flux_rho
    )
    # No core density beyond the last closed flux surface, matching the likelihood.
    core_density = jnp.where(flux_rho <= 1.0, electron_density, 0.0)
    return interferometer_utils.line_integrated_density_signal(
        core_density, sample_points
    )
