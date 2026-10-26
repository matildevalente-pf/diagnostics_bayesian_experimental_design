from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
)


def unpack_flat_target(
    basis: surrogate_types.EquilibriumBasisSpecification,
    y: Float[NpOrJaxArray, " n_y"],
) -> surrogate_types.EquilibriumPrediction:
    """Split one flat target vector into its geometry, iota and flux blocks.

    Args:
        basis: Basis fixing the block widths and the mode tables.
        y: A single flat target vector, laid out as `FLAT_TARGET_LAYOUT`.

    Returns:
        The same content as a prediction object.
    """
    if y.shape != (basis.n_y,):
        raise ValueError(f"y has shape {y.shape} but the basis implies ({basis.n_y},).")
    edges = np.cumsum([basis.n_r, basis.n_z, basis.n_lambda, basis.n_iota])
    return surrogate_types.EquilibriumPrediction(
        coefficients=desc_types.DescFourierZernikeCoefficients(
            r_lmn=y[: edges[0]],
            z_lmn=y[edges[0] : edges[1]],
            lambda_lmn=y[edges[1] : edges[2]],
            r_modes=basis.r_modes,
            z_modes=basis.z_modes,
            lambda_modes=basis.lambda_modes,
            n_field_periods=basis.resolution.n_field_periods,
        ),
        iotaf=y[edges[2] : edges[3]],
        total_toroidal_flux=y[edges[3]],
    )


def flat_target_scale_factors(
    basis: surrogate_types.EquilibriumBasisSpecification,
    machine_scale: surrogate_types.MachineScale,
) -> Float[np.ndarray, " n_y"]:
    """Per-entry factor carrying a normalized flat target to the machine: `L` on the
    `R` and `Z` blocks, one on `lambda` and `iota`, `b L^2` on `Psi`.

    Args:
        basis: Basis fixing the block widths.
        machine_scale: The length and field factors.

    Returns:
        One factor per entry of `y`.
    """
    return np.concatenate(
        [
            np.full(basis.n_r + basis.n_z, machine_scale.length_factor),
            np.ones(basis.n_lambda + basis.n_iota),
            np.array([machine_scale.flux_factor]),
        ]
    )


def scaled_prediction(
    predict: Callable[
        [Float[NpOrJaxArray, "... n_rho"]], Float[NpOrJaxArray, "... n_y"]
    ],
    basis: surrogate_types.EquilibriumBasisSpecification,
    machine_scale: surrogate_types.MachineScale,
    pressure: Float[NpOrJaxArray, "... n_rho"],
) -> Float[jax.Array, "... n_y"]:
    """A normalized surrogate evaluated at the machine: the pressure is divided by `b^2`
    before the model and the flat target scaled by `flat_target_scale_factors` after.
    Bind everything but `pressure` with `functools.partial`.

    Args:
        predict: The normalized forward pass, pressure in Pa at the dataset's field.
        basis: Basis the forward pass predicts in.
        machine_scale: The length and field factors.
        pressure: Pressure profile at the machine, in Pa.

    Returns:
        The flat target at the machine, traceable in `pressure`.
    """
    normalized = predict(jnp.asarray(pressure) / machine_scale.pressure_factor)
    return jnp.asarray(normalized) * jnp.asarray(
        flat_target_scale_factors(basis, machine_scale)
    )


def desc_equilibrium_from_prediction(
    prediction: surrogate_types.EquilibriumPrediction,
    pressure_profile: desc_types.DescSplineProfile,
) -> desc_types.DescEquilibrium:
    """Assemble the DESC equilibrium a prediction describes, under a given pressure.

    The predicted `iotaf` is deliberately not carried over as an iota profile. The
    training targets were solved at prescribed zero net toroidal current, so their
    rotational transform follows from the geometry and `lambda`; prescribing iota
    instead would make a recomputed transform echo the prediction rather than test it.

    Args:
        prediction: Geometry and toroidal flux to build the equilibrium from.
        pressure_profile: Pressure spline the equilibrium was, or would be, solved with.

    Returns:
        An equilibrium at zero net toroidal current, ready for `desc_utils`.
    """
    knots = np.asarray(pressure_profile.knots)
    return desc_types.DescEquilibrium(
        coefficients=prediction.coefficients,
        total_toroidal_flux=prediction.total_toroidal_flux,
        pressure_profile=pressure_profile,
        current_profile=desc_types.DescSplineProfile(
            name="current", knots=knots, params=np.zeros_like(knots)
        ),
    )
