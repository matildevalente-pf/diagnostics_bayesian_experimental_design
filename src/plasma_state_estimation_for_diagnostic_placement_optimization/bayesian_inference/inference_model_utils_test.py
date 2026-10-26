# Tests of the per-volume Thomson calibration corrections the inference model samples:
# 1. One `delta_j` site per active scattering volume, at the configured width.
#    Test: test_one_calibration_site_per_active_volume
# 2. Any subset of active volumes works, and a masked volume keeps `delta_j = 1`.
#    Test: test_corrections_cover_any_subset_of_active_volumes
# 3. The corrections are coordinates of the free-parameter vector, starting at one.
#    Test: test_calibration_sites_are_free_parameters_starting_at_one
#
#   pytest src/.../bayesian_inference/inference_model_utils_test.py


import numpy as np
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_model_utils,
    inference_target_types,
    inference_types,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_types as ts_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_utils as ts,
)

N_VOLUME = 5
N_CHANNEL = len(ts.W7X_POLYCHROMATOR_BANDS)


def _thomson_target(
    active: tuple[int, ...],
) -> inference_target_types.ThomsonSignalTarget:
    """A Thomson target of `N_VOLUME` volumes, only `active` carrying a sigma."""
    theta = np.radians(150.0)
    standoff = ts.collection_object_distance(
        ts.W7X_COLLECTION_FOCAL_LENGTH, ts.W7X_COLLECTION_MAGNIFICATION
    )
    geometry = ts_types.ThomsonScatteringGeometry(
        laser_origin=np.array([-1.0, 0.0, 0.0]),
        laser_direction=np.array([1.0, 0.0, 0.0]),
        reference_phi=0.0,
        volume_positions=np.zeros((N_VOLUME, 3)),
        optic_positions=standoff * np.array([[np.cos(theta), np.sin(theta), 0.0]]),
        volume_optic_index=np.zeros(N_VOLUME, dtype=np.int64),
        collection_aperture_diameter=ts.W7X_COLLECTION_APERTURE_DIAMETER,
        collection_focal_length=ts.W7X_COLLECTION_FOCAL_LENGTH,
        fiber_core_diameter=ts.W7X_FIBER_BUNDLE_DIAMETER,
    )
    uncertainties = np.zeros((N_VOLUME, N_CHANNEL))
    uncertainties[list(active), :] = 1.0e3
    return inference_target_types.ThomsonSignalTarget(
        geometry=geometry,
        response=ts.build_polychromator_response(
            list(ts.W7X_POLYCHROMATOR_BANDS),
            ts.W7X_LASER_WAVELENGTH,
            ts.W7X_LASER_PULSE_ENERGY,
        ),
        signals=np.full((N_VOLUME, N_CHANNEL), 1.0e5),
        uncertainties=uncertainties,
        global_calibration=np.asarray(ts.absolute_calibration(geometry)),
    )


@pytest.mark.parametrize("active", [(0,), (1, 4), (0, 1, 2, 3, 4)])
def test_corrections_cover_any_subset_of_active_volumes(
    active: tuple[int, ...],
) -> None:
    target = _thomson_target(active)
    settings = inference_types.PlasmaEstimationSettings()
    assert inference_model_utils.active_thomson_volumes(target) == active
    priors = inference_model_utils.priors_for(settings, target)
    point = {
        name: np.asarray(0.5 + 0.1 * index)
        for index, name in enumerate(
            priors_utils.site_names(priors, priors_utils.CALIBRATION_PREFIX)
        )
    }
    corrections = np.asarray(
        inference_model_utils.calibration_corrections(point, target)
    )
    expected = np.ones(N_VOLUME)
    expected[list(active)] = [0.5 + 0.1 * index for index in range(len(active))]
    assert corrections.shape == (N_VOLUME,)
    np.testing.assert_allclose(corrections, expected)
    # Without Thomson data there is nothing to correct.
    assert inference_model_utils.calibration_corrections(point, None) is None
    assert inference_model_utils.active_thomson_volumes(None) == ()
