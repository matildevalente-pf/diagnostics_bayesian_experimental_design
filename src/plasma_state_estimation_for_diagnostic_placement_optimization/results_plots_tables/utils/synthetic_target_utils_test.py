# Tests of the calibration the synthetic Thomson data are generated with:
# 1. The ground truth uses the nominal geometric calibration, `delta_j,true = 1`.
#    Test: test_synthetic_truth_uses_the_nominal_geometric_calibration
# 2. Prescribed corrections scale every channel of their own volume.
#    Test: test_prescribed_corrections_scale_their_own_volume
#
#   pytest src/.../results_plots_tables/utils/synthetic_target_utils_test.py

import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
    interferometer_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_types as ts_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_utils as ts,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E501
    synthetic_target_utils,
)

MAJOR_RADIUS = 3.0
VOLUME_RADII = (2.5, 3.0, 3.5)


def _truth() -> desc_truth_types.DescTruth:
    """A circular-cross-section torus with two-power profiles, solved in one step."""
    coefficients = desc_types.DescFourierZernikeCoefficients(
        r_lmn=np.array([MAJOR_RADIUS, 1.0]),
        z_lmn=np.array([1.0]),
        lambda_lmn=np.array([0.0]),
        r_modes=np.array([[0, 0, 0], [1, 1, 0]]),
        z_modes=np.array([[1, -1, 0]]),
        lambda_modes=np.array([[1, -1, 0]]),
        n_field_periods=1,
    )
    return desc_truth_types.DescTruth(
        name="toy",
        coefficients=coefficients,
        profiles=profile_parametrization_types.PlasmaProfiles(
            family="two_power_zero_edge",
            electron_density_params=np.array([3.0, 2.0, 1.0]),
            electron_temperature_params=np.array([2.0, 2.0, 1.0]),
        ),
    )


def _placement() -> diagnostic_design_point.DiagnosticPlacement:
    """Three volumes on the horizontal laser, one lens, one vertical chord."""
    volumes = np.zeros((len(VOLUME_RADII), 3))
    volumes[:, 0] = VOLUME_RADII
    geometry = ts_types.ThomsonScatteringGeometry(
        laser_origin=np.array([2.0, 0.0, 0.0]),
        laser_direction=np.array([1.0, 0.0, 0.0]),
        reference_phi=0.0,
        volume_positions=volumes,
        optic_positions=np.array([[MAJOR_RADIUS, -2.0, 0.0]]),
        volume_optic_index=np.zeros(len(VOLUME_RADII), dtype=np.int64),
        collection_aperture_diameter=ts.W7X_COLLECTION_APERTURE_DIAMETER,
        collection_focal_length=ts.W7X_COLLECTION_FOCAL_LENGTH,
        fiber_core_diameter=ts.W7X_FIBER_BUNDLE_DIAMETER,
    )
    response = ts.build_polychromator_response(
        list(ts.W7X_POLYCHROMATOR_BANDS),
        ts.W7X_LASER_WAVELENGTH,
        ts.W7X_LASER_PULSE_ENERGY,
    )
    return diagnostic_design_point.DiagnosticPlacement(
        design=diagnostic_design_point.DiagnosticDesignPoint(),
        reference_equilibrium_id="toy",
        thomson_geometry=geometry,
        polychromator_response=response,
        thomson_signal_scale=ts.electronic_signal_scale(
            geometry, response, ts.absolute_calibration(geometry)
        ),
        interferometer_chords=[
            interferometer_types.InterferometerGeometry(
                chord_start=np.array([MAJOR_RADIUS, 0.0, -0.9]),
                chord_end=np.array([MAJOR_RADIUS, 0.0, 0.9]),
            )
        ],
    )


def _target(
    true_calibration_factors: np.ndarray | None,
) -> np.ndarray:
    """The noise-free Thomson signals one set of prescribed corrections produces."""
    target = synthetic_target_utils.build_synthetic_target(
        _truth(),
        _placement(),
        interferometer_n_sample=8,
        trim_chords_to_plasma=False,
        true_calibration_factors=true_calibration_factors,
        add_noise=False,
    )
    assert target.thomson_signals is not None
    stored = target.thomson_signals.true_calibration_factors
    if true_calibration_factors is None:
        assert stored is None
    else:
        np.testing.assert_array_equal(np.asarray(stored), true_calibration_factors)
    return np.asarray(target.thomson_signals.signals)


def test_synthetic_truth_uses_the_nominal_geometric_calibration() -> None:
    nominal = _target(None)
    assert np.all(nominal > 0.0)
    np.testing.assert_array_equal(nominal, _target(np.ones(len(VOLUME_RADII))))


def test_prescribed_corrections_scale_their_own_volume() -> None:
    corrections = np.array([0.8, 1.0, 1.25])
    np.testing.assert_allclose(
        _target(corrections), corrections[:, None] * _target(None), rtol=1e-12
    )
