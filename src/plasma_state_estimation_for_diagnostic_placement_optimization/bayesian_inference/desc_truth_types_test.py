# Test of a truth read at the machine scale through the scaled surrogate:
# 1. Its pressure, fed through the wrapper, is the stored spline times `b^2`, and the
#    prediction reproduces the row's equilibrium carried to the machine with the flux
#    `b L^2 Psi`.
#    Test: test_scaled_truth_reproduces_its_equilibrium_and_flux
#
#   pytest src/.../bayesian_inference/desc_truth_types_test.py -m integration

import pathlib

import jax.numpy as jnp
import numpy as np
import pyarrow.compute
import pyarrow.parquet
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    inference_types,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    model_storage_utils,
    surrogate_utils,
)

pytestmark = pytest.mark.integration

_DATA_DIRECTORY = pathlib.Path(__file__).resolve().parents[3] / "data"
_MODELS_DIRECTORY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "mhd_equilibrium_forward_models"
    / "trained_models"
)
_DATASET = "equilibrium_a_3k_spline_input"
_MINOR_RADIUS_NORMALIZED = 0.135


def test_scaled_truth_reproduces_its_equilibrium_and_flux() -> None:
    parquet_path = _DATA_DIRECTORY / f"{_DATASET}.parquet"
    settings_path = _DATA_DIRECTORY / f"{_DATASET}_settings.json"
    model_path = _MODELS_DIRECTORY / f"{inference_types.DEFAULT_SURROGATE_NAME}.npz"
    if not (parquet_path.exists() and model_path.exists()):
        pytest.skip(
            "needs the training parquet and the default surrogate; run "
            "mhd_equilibrium_forward_models/download_from_hugging_face.py"
        )
    scale = diagnostic_design_point.MACHINE_SCALE

    truth = desc_truth_types.read_desc_truth(
        "wide_high_beta_gp", parquet_path, settings_path
    )
    surrogate = model_storage_utils.load_surrogate(
        _MODELS_DIRECTORY,
        inference_types.DEFAULT_SURROGATE_NAME,
        _DATA_DIRECTORY,
        machine_scale=scale,
    )
    assert truth.machine_scale == surrogate.machine_scale == scale
    row = pyarrow.parquet.read_table(
        parquet_path,
        columns=["R_lmn", "Z_lmn", "Psi", "pressure_knots", "pressure_params"],
        filters=pyarrow.compute.field("desc_equilibrium_id")
        == desc_truth_types.DESC_TRUTH_IDS["wide_high_beta_gp"],
    ).to_pylist()[0]
    stored_r = np.asarray(row["R_lmn"])
    stored_z = np.asarray(row["Z_lmn"])
    knots = np.asarray(row["pressure_knots"])
    params = np.asarray(row["pressure_params"])
    grid = np.asarray(surrogate.basis.pressure_rho_grid)
    np.testing.assert_allclose(knots, grid)

    # The truth's geometry is the stored one carried to the machine.
    np.testing.assert_allclose(
        truth.coefficients.r_lmn, scale.length_factor * stored_r, rtol=1e-15
    )
    np.testing.assert_allclose(
        truth.coefficients.z_lmn, scale.length_factor * stored_z, rtol=1e-15
    )

    # The pressure the inference hands the wrapper is the stored spline times b^2.
    pressure = np.asarray(
        profile_parametrization_types.pressure_pascal(truth.profiles, grid)
    )
    inside = knots < 1.0
    np.testing.assert_allclose(
        pressure[inside],
        np.maximum(params[inside] * scale.pressure_factor, 1.0),
        rtol=1e-12,
    )

    # Through the wrapper, the surrogate reproduces the stored solve at the machine to
    # its own accuracy, and the flux exactly up to the model's own error.
    prediction = surrogate_utils.unpack_flat_target(
        surrogate.basis, jnp.asarray(surrogate.predict(jnp.asarray(pressure)))
    )
    tolerance = 0.02 * _MINOR_RADIUS_NORMALIZED * scale.length_factor
    np.testing.assert_allclose(
        prediction.coefficients.r_lmn, scale.length_factor * stored_r, atol=tolerance
    )
    np.testing.assert_allclose(
        prediction.coefficients.z_lmn, scale.length_factor * stored_z, atol=tolerance
    )
    np.testing.assert_allclose(
        float(prediction.total_toroidal_flux),
        scale.flux_factor * row["Psi"],
        rtol=1e-6,
    )
