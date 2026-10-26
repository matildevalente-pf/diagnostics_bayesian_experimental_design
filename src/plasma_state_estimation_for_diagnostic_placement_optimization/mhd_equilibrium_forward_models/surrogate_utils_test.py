# Tests of the machine-scale wrapper around a normalized surrogate:
# 1. Unit factors leave the prediction and the pressure the model sees untouched.
#    Test: test_unit_machine_scale_is_the_identity
# 2. `R` and `Z` scale with `L`, `lambda` and `iota` not at all, `Psi` with `b L^2`,
#    and the model sees the input pressure over `b^2`.
#    Test: test_scaling_laws_on_a_synthetic_prediction
#
#   pytest src/.../mhd_equilibrium_forward_models/surrogate_utils_test.py

import functools

import numpy as np
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
    surrogate_utils,
    testing_utils,
)


def _recording_predict(
    seen: list[np.ndarray],
    y: Float[np.ndarray, " n_y"],
    pressure: Float[NpOrJaxArray, " n_rho"],
) -> Float[np.ndarray, " n_y"]:
    """A stand-in forward pass: records the pressure it is handed and returns `y`."""
    seen.append(np.asarray(pressure))
    return y


def _synthetic_case() -> tuple[
    surrogate_types.EquilibriumBasisSpecification,
    Float[np.ndarray, " n_y"],
    Float[np.ndarray, " n_rho"],
]:
    basis = testing_utils.tiny_basis()
    generator = np.random.default_rng(0)
    return (
        basis,
        generator.normal(size=basis.n_y),
        generator.uniform(1.0e3, 3.0e4, size=basis.n_rho),
    )


def test_unit_machine_scale_is_the_identity() -> None:
    basis, y, pressure = _synthetic_case()
    seen: list[np.ndarray] = []
    predict = functools.partial(
        surrogate_utils.scaled_prediction,
        functools.partial(_recording_predict, seen, y),
        basis,
        surrogate_types.MachineScale(),
    )
    np.testing.assert_array_equal(np.asarray(predict(pressure)), y)
    np.testing.assert_array_equal(seen[0], pressure)


def test_scaling_laws_on_a_synthetic_prediction() -> None:
    basis, y, pressure = _synthetic_case()
    scale = surrogate_types.MachineScale(length_factor=5.5, field_factor=9.0)
    seen: list[np.ndarray] = []
    predict = functools.partial(
        surrogate_utils.scaled_prediction,
        functools.partial(_recording_predict, seen, y),
        basis,
        scale,
    )
    scaled = surrogate_utils.unpack_flat_target(basis, np.asarray(predict(pressure)))
    normalized = surrogate_utils.unpack_flat_target(basis, y)

    np.testing.assert_allclose(seen[0], pressure / 81.0, rtol=1e-15)
    np.testing.assert_allclose(
        scaled.coefficients.r_lmn, 5.5 * normalized.coefficients.r_lmn, rtol=1e-15
    )
    np.testing.assert_allclose(
        scaled.coefficients.z_lmn, 5.5 * normalized.coefficients.z_lmn, rtol=1e-15
    )
    np.testing.assert_array_equal(
        scaled.coefficients.lambda_lmn, normalized.coefficients.lambda_lmn
    )
    np.testing.assert_array_equal(scaled.iotaf, normalized.iotaf)
    np.testing.assert_allclose(
        scaled.total_toroidal_flux,
        9.0 * 5.5**2 * normalized.total_toroidal_flux,
        rtol=1e-15,
    )
