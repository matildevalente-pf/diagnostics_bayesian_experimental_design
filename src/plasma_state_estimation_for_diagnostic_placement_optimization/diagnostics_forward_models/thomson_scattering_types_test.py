import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_types as ts_types,
)


def test_geometry_round_trips() -> None:
    geometry = ts_types.ThomsonScatteringGeometry(
        laser_origin=np.zeros(3),
        laser_direction=np.array([1.0, 0.0, 0.0]),
        reference_phi=0.3,
        volume_positions=np.arange(9.0).reshape(3, 3),
        optic_positions=np.ones((2, 3)),
        volume_optic_index=np.array([0, 1, 0]),
        collection_aperture_diameter=0.16,
        collection_focal_length=0.172,
        fiber_core_diameter=3e-3,
    )
    restored = ts_types.ThomsonScatteringGeometry.model_validate_json(
        geometry.model_dump_json()
    )
    np.testing.assert_allclose(
        np.asarray(restored.volume_positions), np.asarray(geometry.volume_positions)
    )
    assert np.asarray(restored.volume_optic_index).tolist() == [0, 1, 0]
    assert restored.collection_focal_length == 0.172


def test_polychromator_response_round_trips() -> None:
    response = ts_types.PolychromatorResponse(
        laser_wavelength=1064e-9,
        laser_pulse_energy=2.4,
        filter_centers=np.linspace(1.1e-6, 1.4e-6, 5),
        filter_widths=np.full(5, 4e-8),
        band_scales=np.array([0.5, 1.0, 1.1]),
    )
    restored = ts_types.PolychromatorResponse.model_validate_json(
        response.model_dump_json()
    )
    assert restored.laser_wavelength == response.laser_wavelength
    assert np.asarray(restored.filter_centers).shape == (5,)
    np.testing.assert_allclose(np.asarray(restored.band_scales), [0.5, 1.0, 1.1])
