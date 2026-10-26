import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate
import scipy.special
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_types as ts_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_utils as ts,
)

LASER_WAVELENGTH = ts.W7X_LASER_WAVELENGTH

_SIGNAL_SCALE = 1e-12
"""A `kappa` of the order the reference configuration gives, so the floor binds under
about 160 photoelectrons and the readings below are well clear of it."""

SELDEN_1980_TABLE_1: dict[float, float] = {
    0.6: 0.0452,
    0.5: 0.0819,
    0.4: 0.1458,
    0.3: 0.2532,
    0.2: 0.4238,
    0.1: 0.6745,
    0.0: 1.0000,
    -0.1: 1.3433,
    -0.2: 1.5703,
    -0.3: 1.5049,
    -0.4: 1.0782,
    -0.5: 0.4959,
    -0.6: 0.1113,
    -0.7: 0.0069,
}
"""`S_Zh(eps) / S_Zh(0)` at 20 keV and 90 degrees, Selden 1980, Phys. Lett. A 79 405,
table 1, https://doi.org/10.1016/0375-9601(80)90276-5."""


def _geometry(
    standoff_m: float, theta_deg: float, n_volume: int = 1
) -> ts_types.ThomsonScatteringGeometry:
    """`n_volume` coincident volumes viewed by one W7-X lens at `standoff_m`."""
    theta = np.radians(theta_deg)
    return ts_types.ThomsonScatteringGeometry(
        laser_origin=np.array([-1.0, 0.0, 0.0]),
        laser_direction=np.array([1.0, 0.0, 0.0]),
        reference_phi=0.0,
        volume_positions=np.zeros((n_volume, 3)),
        optic_positions=standoff_m * np.array([[np.cos(theta), np.sin(theta), 0.0]]),
        volume_optic_index=np.zeros(n_volume, dtype=np.int64),
        collection_aperture_diameter=ts.W7X_COLLECTION_APERTURE_DIAMETER,
        collection_focal_length=ts.W7X_COLLECTION_FOCAL_LENGTH,
        fiber_core_diameter=ts.W7X_FIBER_BUNDLE_DIAMETER,
    )


def _response() -> ts_types.PolychromatorResponse:
    return ts.build_polychromator_response(
        list(ts.W7X_POLYCHROMATOR_BANDS), LASER_WAVELENGTH, ts.W7X_LASER_PULSE_ENERGY
    )


def _spread_geometry(theta_deg: float) -> ts_types.ThomsonScatteringGeometry:
    """Three volumes at different points of the laser, so each has its own `C^geom`."""
    theta = np.radians(theta_deg)
    standoff = _design_standoff()
    volumes = np.zeros((3, 3))
    volumes[:, 0] = [-0.2, 0.0, 0.2]
    return ts_types.ThomsonScatteringGeometry(
        laser_origin=np.array([-1.0, 0.0, 0.0]),
        laser_direction=np.array([1.0, 0.0, 0.0]),
        reference_phi=0.0,
        volume_positions=volumes,
        optic_positions=standoff * np.array([[np.cos(theta), np.sin(theta), 0.0]]),
        volume_optic_index=np.zeros(3, dtype=np.int64),
        collection_aperture_diameter=ts.W7X_COLLECTION_APERTURE_DIAMETER,
        collection_focal_length=ts.W7X_COLLECTION_FOCAL_LENGTH,
        fiber_core_diameter=ts.W7X_FIBER_BUNDLE_DIAMETER,
    )


def _design_standoff() -> float:
    return ts.collection_object_distance(
        ts.W7X_COLLECTION_FOCAL_LENGTH, ts.W7X_COLLECTION_MAGNIFICATION
    )


def test_zhuravlev_spectrum_matches_selden_1980_table_1() -> None:
    two_alpha = ts.ELECTRON_REST_ENERGY_EV / 20e3
    shifts = jnp.array(list(SELDEN_1980_TABLE_1))
    spectrum = ts._zhuravlev_spectral_density(shifts, np.pi / 2.0, two_alpha)
    peak = ts._zhuravlev_spectral_density(jnp.array([0.0]), np.pi / 2.0, two_alpha)[0]
    np.testing.assert_allclose(
        np.asarray(spectrum / peak), list(SELDEN_1980_TABLE_1.values()), atol=1e-4
    )


@pytest.mark.parametrize("te_ev", [1e3, 10e3, 50e3, 100e3])
def test_depolarization_matches_naito_exact_backscatter_form(te_ev: float) -> None:
    # Naito 1993 p. 4256: at 180 degrees the depolarization factor is exactly
    # q = 1 + 2 exp(2 alpha x) [E_3(2 alpha x) - 3 E_5(2 alpha x)], and the (2,2)
    # rational form is claimed within 0.1% up to 100 keV. The original paper's sign of
    # the zeta^6 term of p2, corrected in the erratum, misses this by 15% at 100 keV.
    two_alpha = ts.ELECTRON_REST_ENERGY_EV / te_ev
    shifts = jnp.linspace(-0.6, 0.5, 23)
    x, _, _ = ts._naito_variables(shifts, np.pi)
    argument = np.asarray(two_alpha * x)
    exact = 1.0 + 2.0 * np.exp(argument) * (
        scipy.special.expn(3, argument) - 3.0 * scipy.special.expn(5, argument)
    )
    approximate = np.asarray(ts._depolarization_factor(shifts, np.pi, two_alpha))
    np.testing.assert_allclose(approximate, exact, rtol=1e-3)


@pytest.mark.parametrize("theta_deg", [30.0, 150.0])
def test_cold_spectrum_is_a_unit_normalized_gaussian(theta_deg: float) -> None:
    # Hutchinson 2002, eqs. 7.2.23-7.2.25 (section 7.2.3): the nonrelativistic
    # spectrum is a Gaussian, of width 2 sin(theta / 2) sqrt(Te / m c^2) in eps (cf.
    # eq. 7.2.35, section 7.2.4) and unit integral.
    shifts = np.linspace(-0.05, 0.05, 100001)
    spectrum = np.asarray(
        ts.spectral_density_relativistic(
            jnp.asarray(LASER_WAVELENGTH * (1.0 + shifts)),
            np.radians(theta_deg),
            20.0,
            LASER_WAVELENGTH,
        )
    )
    width = (
        2.0
        * np.sin(np.radians(theta_deg) / 2.0)
        * np.sqrt(20.0 / ts.ELECTRON_REST_ENERGY_EV)
    )
    gaussian = np.exp(-0.5 * (shifts / width) ** 2) / (width * np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(np.trapezoid(spectrum, shifts), 1.0, atol=1e-3)
    assert np.max(np.abs(spectrum - gaussian)) / gaussian.max() < 0.03


def test_collection_optics_reproduce_pasch_2016_volume_lengths() -> None:
    # Pasch 2016 section II: 25 mm volumes at 7.8:1 demagnification, 37 mm at 11.5:1.
    for magnification, expected_mm in ((1.0 / 7.8, 25.0), (1.0 / 11.5, 37.0)):
        standoff = ts.collection_object_distance(
            ts.W7X_COLLECTION_FOCAL_LENGTH, magnification
        )
        volume_length, solid_angle = ts.collection_optics(_geometry(standoff, 90.0))
        np.testing.assert_allclose(float(volume_length[0]) * 1e3, expected_mm, rtol=0.1)
        # A few 1e-3 sr, Hutchinson 2002 section 7.2.4 ("perhaps 1e-2 sr").
        assert 4e-3 <= float(solid_angle[0]) <= 1e-2


def test_collection_optics_reject_a_standoff_inside_the_focal_length() -> None:
    with pytest.raises(ValueError, match="focal length"):
        ts.collection_optics(_geometry(0.16, 30.0))


def test_signals_match_the_bozhenkov_integrand_quadrature() -> None:
    # Bozhenkov 2017 eq. (1.2): D_i = dL dOmega N_L n_e r_e^2 int (1 + eps) S deps
    # over the passband, evaluated here with an adaptive quadrature.
    geometry = _geometry(_design_standoff(), 30.0)
    response = _response()
    density, temperature = 1e19, 1e3
    calibration = float(ts.absolute_calibration(geometry)[0])
    signals = np.asarray(
        ts.thomson_signals(
            geometry,
            response,
            jnp.array([density]),
            jnp.array([temperature]),
            calibration,
        )
    )[0]
    laser_photons = ts.W7X_LASER_PULSE_ENERGY * LASER_WAVELENGTH / ts.PLANCK_TIMES_C
    for channel, (low, high) in enumerate(ts.W7X_POLYCHROMATOR_BANDS):
        integral, _ = scipy.integrate.quad(
            lambda eps: (
                (1.0 + eps)
                * float(
                    ts.spectral_density_relativistic(
                        jnp.array(LASER_WAVELENGTH * (1.0 + eps)),
                        np.radians(30.0),
                        temperature,
                        LASER_WAVELENGTH,
                    )
                )
            ),
            low / LASER_WAVELENGTH - 1.0,
            high / LASER_WAVELENGTH - 1.0,
            epsabs=0.0,
            epsrel=1e-10,
            limit=200,
        )
        expected = (
            calibration
            * laser_photons
            * density
            * ts.CLASSICAL_ELECTRON_RADIUS**2
            * integral
        )
        np.testing.assert_allclose(signals[channel], expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("theta_deg", "te_ev"), [(30.0, 20.0), (150.0, 20.0), (30.0, 1e4), (150.0, 1e4)]
)
def test_signals_are_converged_in_quadrature_nodes(
    theta_deg: float, te_ev: float
) -> None:
    geometry = _geometry(_design_standoff(), theta_deg)
    response = _response()
    signals = np.asarray(
        ts.thomson_signals(
            geometry, response, jnp.array([1e19]), jnp.array([te_ev]), 1.0
        )
    )[0]
    nodes, weights = np.polynomial.legendre.leggauss(4 * ts.BAND_QUADRATURE_NODES)
    reference = []
    for low, high in ts.W7X_POLYCHROMATOR_BANDS:
        eps_low, eps_high = low / LASER_WAVELENGTH - 1.0, high / LASER_WAVELENGTH - 1.0
        eps = 0.5 * (eps_high - eps_low) * nodes + 0.5 * (eps_high + eps_low)
        spectrum = np.asarray(
            ts.spectral_density_relativistic(
                jnp.asarray(LASER_WAVELENGTH * (1.0 + eps)),
                np.radians(theta_deg),
                te_ev,
                LASER_WAVELENGTH,
            )
        )
        reference.append(
            0.5 * (eps_high - eps_low) * np.sum(weights * (1.0 + eps) * spectrum)
        )
    laser_photons = ts.W7X_LASER_PULSE_ENERGY * LASER_WAVELENGTH / ts.PLANCK_TIMES_C
    reference = (
        laser_photons * 1e19 * ts.CLASSICAL_ELECTRON_RADIUS**2 * np.asarray(reference)
    )
    lit = reference > 1e-6 * reference.sum()
    np.testing.assert_allclose(signals[lit], reference[lit], rtol=1e-6)


def test_signals_are_linear_in_density_and_hotter_volumes_reach_outer_channels() -> (
    None
):
    geometry = _geometry(_design_standoff(), 90.0, n_volume=2)
    response = _response()
    temperatures = jnp.array([1e3, 4e3])
    signals = ts.thomson_signals(
        geometry, response, jnp.array([5e19, 5e19]), temperatures, 1.0
    )
    doubled = ts.thomson_signals(
        geometry, response, jnp.array([1e20, 1e20]), temperatures, 1.0
    )
    np.testing.assert_allclose(
        np.asarray(doubled), 2.0 * np.asarray(signals), rtol=1e-12
    )
    outer = np.asarray(signals[:, 0] / signals.sum(axis=1))
    assert outer[1] > outer[0]


def test_band_edges_scale_every_distance_to_the_laser_line() -> None:
    response = _response()
    low, high = ts.band_edges(response)
    scaled_low, scaled_high = ts.band_edges(response, jnp.array([0.5, 1.0]))
    np.testing.assert_allclose(np.asarray(scaled_low[1]), np.asarray(low))
    np.testing.assert_allclose(
        LASER_WAVELENGTH - np.asarray(scaled_low[0]),
        0.5 * (LASER_WAVELENGTH - np.asarray(low)),
    )
    np.testing.assert_allclose(
        np.asarray(scaled_high[0] - scaled_low[0]), 0.5 * np.asarray(high - low)
    )


def test_band_scale_bounds_enforce_width_and_edge_constraints() -> None:
    response = _response()
    low, high = ts.band_scale_bounds(response)
    low_edges, high_edges = ts.band_edges(response, jnp.array([low, high]))
    np.testing.assert_allclose(
        float(jnp.min(high_edges[0] - low_edges[0])), 5e-9, rtol=1e-12
    )
    np.testing.assert_allclose(float(jnp.min(low_edges[1])), 700e-9, rtol=1e-12)
    assert low < 1.0 < high


def test_poisson_uncertainty_is_the_square_root_of_the_signal() -> None:
    signals = jnp.array([[100.0, 5.0]])
    sigma = np.asarray(ts.signal_uncertainties(signals, "poisson", _SIGNAL_SCALE))
    np.testing.assert_allclose(sigma[0, 0], 10.0)
    # Only the counting model keeps the faint-channel mask: sqrt(D) is no usable width
    # below ten photoelectrons, whereas the electronic models have their floor.
    assert sigma[0, 1] == 0.0


@pytest.mark.parametrize("signal", [1e3, 1e4, 1e5])
def test_electronic_uncertainty_follows_the_raak_curve_above_the_floor(
    signal: float,
) -> None:
    sigma = float(
        ts.signal_uncertainties(jnp.array([[signal]]), "electronic", _SIGNAL_SCALE)[
            0, 0
        ]
    )
    relative = (
        ts.ELECTRONIC_NOISE_A * np.exp(-ts.ELECTRONIC_NOISE_B * _SIGNAL_SCALE * signal)
        + ts.ELECTRONIC_NOISE_C
    )
    assert sigma > _noise_floor()
    np.testing.assert_allclose(sigma, signal * relative, rtol=1e-12)


def test_a_vanishing_signal_keeps_the_electronic_noise_floor() -> None:
    signals = jnp.array([[0.0, 1e-3, 1.0]])
    sigma = ts.signal_uncertainties(signals, "electronic", _SIGNAL_SCALE)
    # sigma -> sigma_floor and SNR -> 0, rather than sigma -> 0 and a hard constraint.
    np.testing.assert_allclose(np.asarray(sigma), _noise_floor(), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(signals / sigma)[0, 0], 0.0)


def test_the_signal_scale_is_the_reference_second_channel_at_snr_two() -> None:
    geometry = _geometry(_design_standoff(), 155.0, n_volume=3)
    response = _response()
    calibration = ts.absolute_calibration(geometry)
    reference = np.asarray(
        ts.thomson_signals(
            geometry,
            response,
            np.full(3, ts.MINIMUM_DETECTABLE_DENSITY),
            np.full(3, ts.MINIMUM_DETECTABLE_TEMPERATURE_EV),
            calibration,
        )
    )
    # Raak 2024 needs two channels to fit a spectrum, so the second largest reading of
    # the weakest polychromator is what the detector has to resolve.
    required = float(np.min(np.sort(reference, axis=1)[:, -2]))
    signal_scale = ts.electronic_signal_scale(geometry, response, calibration)
    np.testing.assert_allclose(
        signal_scale,
        ts.MINIMUM_DETECTABLE_INTEGRATED_SIGNAL / required,
        rtol=1e-12,
    )
    sigma = ts.signal_uncertainties(jnp.array([[required]]), "electronic", signal_scale)
    np.testing.assert_allclose(required / float(sigma[0, 0]), 2.0, rtol=1e-12)


def test_a_distant_optic_loses_snr_at_the_reference_signal_scale() -> None:
    response = _response()
    reference_geometry = _geometry(_design_standoff(), 155.0)
    signal_scale = ts.electronic_signal_scale(
        reference_geometry, response, ts.absolute_calibration(reference_geometry)
    )
    far_geometry = _geometry(2.0 * _design_standoff(), 155.0)
    ratios = []
    for geometry in (reference_geometry, far_geometry):
        signals = ts.thomson_signals(
            geometry,
            response,
            jnp.array([2e19]),
            jnp.array([1e3]),
            ts.absolute_calibration(geometry),
        )
        sigma = ts.signal_uncertainties(signals, "electronic", signal_scale)
        ratios.append(np.asarray(signals / sigma)[0])
    # Less light through the same detector, so every channel loses signal to noise.
    assert np.all(ratios[1] < ratios[0])
    # The penalty only exists because `kappa` stayed put: recalibrating it on the far
    # layout would hand that layout back most of its ratio.
    far_scale = ts.electronic_signal_scale(
        far_geometry, response, ts.absolute_calibration(far_geometry)
    )
    assert far_scale > 1.5 * signal_scale


def test_volume_scattering_angles_match_geometry() -> None:
    angles = ts.volume_scattering_angles(_geometry(1.0, 90.0))
    np.testing.assert_allclose(np.asarray(angles), np.pi / 2.0, atol=1e-12)


def test_unit_corrections_reproduce_the_geometric_calibration() -> None:
    geometry = _spread_geometry(150.0)
    response = _response()
    nominal = ts.absolute_calibration(geometry)
    density = jnp.array([4e19, 3e19, 2e19])
    temperature = jnp.array([1e3, 2e3, 3e3])
    reference = ts.thomson_signals(geometry, response, density, temperature, nominal)
    for corrections in (None, jnp.ones(3)):
        np.testing.assert_array_equal(
            np.asarray(ts.effective_calibration(nominal, corrections)),
            np.asarray(nominal),
        )
        signals = ts.thomson_signals(
            geometry,
            response,
            density,
            temperature,
            ts.effective_calibration(nominal, corrections),
        )
        np.testing.assert_array_equal(np.asarray(signals), np.asarray(reference))


def test_each_correction_scales_only_its_own_polychromator() -> None:
    geometry = _spread_geometry(150.0)
    response = _response()
    nominal = ts.absolute_calibration(geometry)
    density = jnp.array([4e19, 3e19, 2e19])
    temperature = jnp.array([1e3, 2e3, 3e3])
    corrections = np.array([0.8, 1.0, 1.25])
    reference = np.asarray(
        ts.thomson_signals(geometry, response, density, temperature, nominal)
    )
    corrected = np.asarray(
        ts.thomson_signals(
            geometry,
            response,
            density,
            temperature,
            ts.effective_calibration(nominal, jnp.asarray(corrections)),
        )
    )
    # One factor per volume, applied to every channel of that volume and to no other.
    np.testing.assert_allclose(corrected, corrections[:, None] * reference, rtol=1e-12)


@pytest.mark.parametrize("n_volume", [1, 2, 7])
def test_effective_calibration_holds_for_any_number_of_volumes(n_volume: int) -> None:
    nominal = jnp.linspace(1e-3, 2e-3, n_volume)
    corrections = jnp.linspace(0.7, 1.3, n_volume)
    np.testing.assert_allclose(
        np.asarray(ts.effective_calibration(nominal, corrections)),
        np.asarray(nominal) * np.asarray(corrections),
        rtol=1e-14,
    )


def _noise_floor() -> float:
    """`sigma_floor` in photoelectrons at `_SIGNAL_SCALE`."""
    return ts.NOISE_FLOOR_INTEGRATED_SIGNAL / _SIGNAL_SCALE
