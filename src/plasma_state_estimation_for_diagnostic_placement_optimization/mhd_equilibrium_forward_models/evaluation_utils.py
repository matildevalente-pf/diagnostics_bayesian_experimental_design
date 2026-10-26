# Scoring a surrogate against the DESC solves it should reproduce: per-sample geometry,
# transform, field and force-residual metrics on grids shared by a whole test set.

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
    surrogate_utils,
)

# Importing `desc` flips JAX to float64 globally, so the runtime import stays inside
# `build_evaluation_grids` and only the annotations resolve it here.
if TYPE_CHECKING:
    import desc.equilibrium
    import desc.grid


class EvaluationGrids(NamedTuple):
    """The DESC grids the metrics are evaluated on, built once for a whole test set."""

    flux_surface: "desc.grid.LinearGrid"
    """Covers the torus for the flux-surface displacement; `rho = 0` is left out."""

    magnetic_axis: "desc.grid.LinearGrid"
    """A toroidal ring at `rho = 0`."""

    iota_radial: "desc.grid.LinearGrid"
    """Uniform radial grid, so the transform RMSE is not weighted by repeated `rho`."""

    iota_predicted: "desc.grid.LinearGrid"
    """The basis' own `iota_rho_grid`, where the predicted `iotaf` block lives."""


class SampleMetrics(NamedTuple):
    """The metrics of one predicted equilibrium against its target."""

    flux_surface_error_m: float
    """Mean distance between the predicted and target flux surfaces [m]."""

    magnetic_axis_error_m: float
    """RMSE between the predicted and target magnetic axes [m]."""

    iota_computed_rmse: float
    """RMSE of the transform recomputed from the predicted geometry."""

    iota_predicted_rmse: float
    """RMSE of the directly predicted `iotaf` block."""

    field_strength_relative_rmse: float
    """RMSE of `|B|` over the flux-surface grid, relative to the target's mean `|B|`.

    The only metric here that reads the field rather than the geometry, so it is what
    first-derivative accuracy buys, one derivative short of the force residual.
    """

    psi_relative_error: float
    """Relative error of the predicted total toroidal flux."""

    fnorm_predicted: float
    """Normalized force residual of the predicted equilibrium."""

    fnorm_target: float
    """Normalized force residual of the target equilibrium."""


def evaluate_sample(
    prediction: surrogate_types.EquilibriumPrediction,
    target: desc_types.DescEquilibrium,
    basis: surrogate_types.EquilibriumBasisSpecification,
    grids: EvaluationGrids,
) -> SampleMetrics:
    """Score one predicted equilibrium against its target.

    The prediction is rebuilt under the target's own pressure spline, so the two DESC
    objects differ only in geometry.

    Args:
        prediction: The surrogate's output for this sample.
        target: The DESC solve it should reproduce.
        basis: Basis both sides are expressed in.
        grids: Grids to evaluate on, from `build_evaluation_grids`.

    Returns:
        The metrics of this sample.
    """
    predicted_equilibrium = desc_utils.desc_equilibrium_to_desc_object(
        surrogate_utils.desc_equilibrium_from_prediction(
            prediction, target.pressure_profile
        ),
        basis.resolution,
    )
    target_equilibrium = desc_utils.desc_equilibrium_to_desc_object(
        target, basis.resolution
    )

    # `|B|` is asked for in the same call as the geometry rather than in a second one,
    # so DESC resolves the transforms both of them depend on once.
    predicted_geometry = predicted_equilibrium.compute(
        ["R", "Z", "|B|"], grid=grids.flux_surface
    )
    target_geometry = target_equilibrium.compute(
        ["R", "Z", "|B|"], grid=grids.flux_surface
    )
    displacement = np.sqrt(
        (np.asarray(predicted_geometry["R"]) - np.asarray(target_geometry["R"])) ** 2
        + (np.asarray(predicted_geometry["Z"]) - np.asarray(target_geometry["Z"])) ** 2
    )
    predicted_field_strength = np.asarray(predicted_geometry["|B|"])
    target_field_strength = np.asarray(target_geometry["|B|"])

    target_psi = float(target.total_toroidal_flux)
    predicted_psi = float(prediction.total_toroidal_flux)
    return SampleMetrics(
        flux_surface_error_m=float(np.mean(displacement)),
        magnetic_axis_error_m=_axis_error(
            predicted_equilibrium, target_equilibrium, grids
        ),
        # Both transforms in absolute value: DESC and VMEC disagree on the sign of
        # `iota`, and a model trained on either convention is equally correct.
        iota_computed_rmse=_rmse(
            np.abs(_iota(predicted_equilibrium, grids.iota_radial)),
            np.abs(_iota(target_equilibrium, grids.iota_radial)),
        ),
        iota_predicted_rmse=_rmse(
            np.abs(np.asarray(prediction.iotaf)),
            np.abs(_iota(target_equilibrium, grids.iota_predicted)),
        ),
        field_strength_relative_rmse=_rmse(
            predicted_field_strength, target_field_strength
        )
        / float(np.mean(target_field_strength)),
        psi_relative_error=abs(predicted_psi - target_psi)
        / max(abs(target_psi), 1e-30),
        fnorm_predicted=force_residual(predicted_equilibrium),
        fnorm_target=force_residual(target_equilibrium),
    )


def build_evaluation_grids(
    basis: surrogate_types.EquilibriumBasisSpecification,
) -> EvaluationGrids:
    """Build the four DESC grids the metrics are evaluated on.

    Args:
        basis: Basis supplying the field-period count, the symmetry and the iota grid.

    Returns:
        The grids, to be shared across every sample of a test set.
    """
    import desc.grid

    n_field_periods = basis.resolution.n_field_periods
    return EvaluationGrids(
        flux_surface=desc.grid.LinearGrid(
            rho=np.linspace(0.1, 1.0, 30),
            theta=np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False),
            N=basis.resolution.toroidal_resolution,
            NFP=n_field_periods,
            sym=False,
        ),
        magnetic_axis=desc.grid.LinearGrid(
            rho=np.array([0.0]),
            N=64,
            NFP=n_field_periods,
            sym=basis.resolution.is_stellarator_symmetric,
        ),
        iota_radial=desc.grid.LinearGrid(
            rho=np.linspace(0.0, 1.0, 30), NFP=n_field_periods
        ),
        iota_predicted=desc.grid.LinearGrid(
            rho=np.asarray(basis.iota_rho_grid), NFP=n_field_periods
        ),
    )


def force_residual(
    equilibrium: "desc.equilibrium.Equilibrium",
) -> float:
    """Compute the volume-averaged normalized force residual of an equilibrium.

    Args:
        equilibrium: A live DESC equilibrium.

    Returns:
        `<|F|>_vol` divided by `<|grad(|B|^2)|/2mu0>_vol`.
    """
    data = equilibrium.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
    return float(data["<|F|>_vol"]) / float(data["<|grad(|B|^2)|/2mu0>_vol"])


def _axis_error(
    predicted_equilibrium: "desc.equilibrium.Equilibrium",
    target_equilibrium: "desc.equilibrium.Equilibrium",
    grids: EvaluationGrids,
) -> float:
    """RMSE between two magnetic axes [m], over `R` and `Z` concatenated."""
    predicted = predicted_equilibrium.compute(["R", "Z"], grid=grids.magnetic_axis)
    target = target_equilibrium.compute(["R", "Z"], grid=grids.magnetic_axis)
    return _rmse(
        np.concatenate([np.asarray(predicted["R"]), np.asarray(predicted["Z"])]),
        np.concatenate([np.asarray(target["R"]), np.asarray(target["Z"])]),
    )


def _iota(
    equilibrium: "desc.equilibrium.Equilibrium",
    grid: "desc.grid.LinearGrid",
) -> Float[NpOrJaxArray, " n_points"]:
    """The rotational transform of an equilibrium at each point of a grid."""
    return np.asarray(equilibrium.compute(["iota"], grid=grid)["iota"])


def _rmse(
    predicted: Float[NpOrJaxArray, " n_points"],
    target: Float[NpOrJaxArray, " n_points"],
) -> float:
    """Root mean square difference between two arrays."""
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(target)) ** 2)))
