# The physics term of the curriculum: DESC's force-balance residual as a training loss.
#   python -m plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models.physics_loss_utils  # noqa: E501

import math
import pathlib
from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
    surrogate_utils,
)

if TYPE_CHECKING:
    import desc.objectives

DescEquilibriumParams = Mapping[str, jax.Array | np.ndarray | float | int | bool | str]
"""A DESC equilibrium's `params_dict`, the parameters its objectives are evaluated at."""

_NORMALIZATION_FLOOR = 1e-30
"""Keeps the residual finite where the toroidal flux vanishes."""


def build_fnorm_loss(
    basis: surrogate_types.EquilibriumBasisSpecification,
    template_y: Float[NpOrJaxArray, " n_y"],
    template_pressure_knot_values: Float[NpOrJaxArray, " n_knots"],
    pressure_knots: Float[NpOrJaxArray, " n_knots"],
    loss_weight: float,
) -> jax.tree_util.Partial:
    """Build the per-sample loss on the force-balance residual of the prediction.

    The term is the logarithm of the prediction's own residual, so it is negative
    throughout. One `ForceBalance` is built here, on the row `template_y` describes,
    and every sample is evaluated by substituting its own coefficients into it.

    Args:
        basis: Basis the predictions and targets are expressed in.
        template_y: Flat target vector of the row the objective is built on.
        template_pressure_knot_values: That row's pressure at the spline knots [Pa].
        pressure_knots: Knots of the stored pressure splines, shared by every row.
        loss_weight: Overall weight of the term.

    Returns:
        A callable `(y_predicted, y_target_with_pressure) -> scalar`, to be vmapped over a
        batch, whose second argument is the target vector with the sample's own pressure
        at the spline knots in Pascals appended.
    """
    import desc.equilibrium
    import desc.grid
    import desc.objectives
    import desc.profiles

    prediction = surrogate_utils.unpack_flat_target(
        basis, np.asarray(template_y, dtype=np.float64)
    )
    coefficients = prediction.coefficients
    resolution = basis.resolution
    # The pressure is the stored spline the rows were solved with, so the term reads the
    # same `dp/drho` as the solver did and as the reported metric does.
    template = desc.equilibrium.Equilibrium(
        R_lmn=np.asarray(coefficients.r_lmn, dtype=np.float64),
        Z_lmn=np.asarray(coefficients.z_lmn, dtype=np.float64),
        L_lmn=np.asarray(coefficients.lambda_lmn, dtype=np.float64),
        L=resolution.radial_resolution,
        M=resolution.poloidal_resolution,
        N=resolution.toroidal_resolution,
        NFP=resolution.n_field_periods,
        sym=resolution.is_stellarator_symmetric,
        spectral_indexing=resolution.spectral_indexing,
        Psi=float(prediction.total_toroidal_flux),
        # The stored coefficients are a converged solution, so DESC must take them as
        # given rather than flipping theta to enforce its own orientation convention.
        check_orientation=False,
        pressure=desc.profiles.SplineProfile(
            values=np.asarray(template_pressure_knot_values, dtype=np.float64),
            knots=np.asarray(pressure_knots, dtype=np.float64),
            method="cubic2",
        ),
    )
    force_balance = desc.objectives.ForceBalance(template, normalize=False)
    force_balance.build(verbose=0)

    # DESC's own force scale `B^2 / (2 mu_0 a) * V` at `B = 1.25 Psi / (pi a^2)`,
    # divided by `Psi^2` so the per-sample flux multiplies back in per evaluation. `a`
    # and `V` are computed from the geometry, not read from DESC's `scales`, whose
    # boundary-mode estimates describe the default surface this template carries:
    # https://github.com/PlasmaControl/DESC/blob/master/desc/objectives/normalization.py
    geometry = template.compute(
        ["a", "V"], grid=desc.grid.LinearGrid(L=2, M=2, N=2, NFP=template.NFP)
    )
    minor_radius = float(geometry["a"])
    mu_0 = 4.0 * math.pi * 1e-7
    normalization_constant = (
        (1.25 / (math.pi * minor_radius**2)) ** 2
        / (2.0 * mu_0 * minor_radius)
        * float(geometry["V"])
    )

    return jax.tree_util.Partial(
        fnorm_loss,
        n_r=basis.n_r,
        n_z=basis.n_z,
        n_geometry=basis.n_r + basis.n_z + basis.n_lambda,
        n_y=basis.n_y,
        normalization_constant=normalization_constant,
        template_params=template.params_dict,
        force_balance=force_balance,
        loss_weight=loss_weight,
    )


def fnorm_loss(
    y_predicted: Float[NpOrJaxArray, " n_y"],
    y_target_with_pressure: Float[NpOrJaxArray, " n_y_and_knots"],
    *,
    n_r: int,
    n_z: int,
    n_geometry: int,
    n_y: int,
    normalization_constant: float,
    template_params: DescEquilibriumParams,
    force_balance: "desc.objectives.ForceBalance",
    loss_weight: float,
) -> Float[jax.Array, ""]:
    """Score one prediction by the logarithm of its own force residual.

    The pressure and the toroidal flux are read from the target, not the prediction, so
    the term is a function of the predicted geometry alone. The residual is computed in
    float64 and cast back, which keeps the training graph float32.

    Args:
        y_predicted: One predicted flat target vector.
        y_target_with_pressure: The target vector with the sample's own pressure at the
            spline knots in Pascals appended, so that a batch is one array.
        n_r: Width of the `R` block.
        n_z: Width of the `Z` block.
        n_geometry: Combined width of the three coefficient blocks.
        n_y: Width of the flat target vector, i.e. where the pressure starts.
        normalization_constant: Geometric constant the residual is divided by.
        template_params: `params_dict` of the equilibrium the objective was built on.
        force_balance: A built `ForceBalance` with DESC's own normalization switched off.
        loss_weight: Overall weight of the term.

    Returns:
        `loss_weight * log(predicted)`, so that `exp(value / loss_weight)` recovers the
        residual itself. Negative throughout, because the residual is far below one.
    """
    pressure_knot_values = y_target_with_pressure[n_y:].astype(jnp.float64)
    predicted = normalized_force_residual(
        y_predicted[:n_r].astype(jnp.float64),
        y_predicted[n_r : n_r + n_z].astype(jnp.float64),
        y_predicted[n_r + n_z : n_geometry].astype(jnp.float64),
        pressure_knot_values,
        y_target_with_pressure[n_y - 1].astype(jnp.float64),
        normalization_constant,
        template_params,
        force_balance,
    )
    return (loss_weight * jnp.log(predicted)).astype(jnp.float32)


def normalized_force_residual(
    r_lmn: Float[NpOrJaxArray, " n_modes_r"],
    z_lmn: Float[NpOrJaxArray, " n_modes_z"],
    lambda_lmn: Float[NpOrJaxArray, " n_modes_lambda"],
    pressure_knot_values: Float[NpOrJaxArray, " n_knots"],
    total_toroidal_flux: Float[NpOrJaxArray, ""],
    normalization_constant: float,
    template_params: DescEquilibriumParams,
    force_balance: "desc.objectives.ForceBalance",
) -> Float[jax.Array, ""]:
    """Evaluate DESC's force-balance objective on a set of coefficients, dimensionlessly.

    The geometry, the flux and the pressure replace their entries in the template's
    parameter dictionary, so one built objective serves every sample of a training run.

    Args:
        r_lmn: Fourier-Zernike coefficients of `R` [m].
        z_lmn: Fourier-Zernike coefficients of `Z` [m].
        lambda_lmn: Fourier-Zernike coefficients of `lambda` [rad].
        pressure_knot_values: Pressure at the spline knots [Pa], a spline's `p_l` in DESC.
        total_toroidal_flux: Toroidal flux through the last closed flux surface [Wb].
        normalization_constant: Geometric constant, as bound by `build_fnorm_loss`.
        template_params: `params_dict` of the equilibrium the objective was built on.
        force_balance: A built `ForceBalance` with DESC's own normalization switched off.

    Returns:
        The root mean square of the objective's per-node forces [N], divided by
        `normalization_constant * total_toroidal_flux ** 2`.
    """
    residual = force_balance.compute_scaled_error(
        {
            **template_params,
            "R_lmn": r_lmn,
            "Z_lmn": z_lmn,
            "L_lmn": lambda_lmn,
            "Psi": total_toroidal_flux,
            "p_l": pressure_knot_values,
        },
        constants=force_balance.constants,
    )
    normalization = (
        jnp.asarray(normalization_constant, dtype=jnp.float64) * total_toroidal_flux**2
        + _NORMALIZATION_FLOOR
    )
    return jnp.sqrt(jnp.mean((residual / normalization) ** 2))


if __name__ == "__main__":
    # Print this residual beside the one each DESC solve stored, on five training rows.
    # Evaluated at the rows' own stored coefficients, so both numbers score the same
    # converged equilibrium. They are different metrics, not one scaled by a constant:
    # this is an RMS over per-node forces [N], the stored one a volume average [N/m^3].
    import pandas as pd
    from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
        dataset_utils,
    )

    # Five training rows.
    CHECK_EQUILIBRIUM_IDS = (
        "DGhCMoV2YHgCQHZo2jSiSzh",
        "DYGurcjmTfc4StJAuuD4aDT",
        "DdPFS5EMrutUqjM8N3C3ehA",
        "D8PrKzkkNYWZyUDGE4miHpf",
        "D8mT7eHQKKLdbTBFqNAHKG7",
    )

    data_directory = pathlib.Path(__file__).parents[3] / "data"
    parquet_path = data_directory / f"{dataset_utils.DEFAULT_DATASET}.parquet"
    metadata = dataset_utils.load_metadata(
        data_directory / f"{dataset_utils.DEFAULT_DATASET}_settings.json"
    )
    dataset = dataset_utils.load_dataset(parquet_path, metadata, max_fnorm_vol=math.inf)
    stored = pd.read_parquet(
        parquet_path,
        columns=["desc_equilibrium_id", "fnorm_volume_averaged_force_error"],
    ).set_index("desc_equilibrium_id")
    rows = [dataset.equilibrium_ids.index(name) for name in CHECK_EQUILIBRIUM_IDS]
    built = build_fnorm_loss(
        metadata.basis,
        dataset.y[rows[0]],
        dataset.pressure_params[rows[0]],
        dataset.pressure_knots[rows[0]],
        loss_weight=1.0,
    )

    print(
        f"{'equilibrium':<25}{'f [N]':>11}{'<|F|>_vol':>11}"
        f"{'fnorm':>11}{'stored':>11}{'ratio':>8}"
    )
    ratios = []
    for row in rows:
        y = dataset.y[row]
        name = dataset.equilibrium_ids[row]
        fnorm = float(
            jnp.exp(built(y, jnp.concatenate([y, dataset.pressure_params[row]])))
        )
        constant = built.keywords["normalization_constant"]
        ratios.append(fnorm / dataset.fnorm_vol[row])
        print(
            f"{name:<25}{fnorm * constant * float(y[-1]) ** 2:>11.3e}"
            f"{stored.loc[name].iloc[0]:>11.3e}"
            f"{fnorm:>11.3e}{dataset.fnorm_vol[row]:>11.3e}{ratios[-1]:>8.3f}"
        )
    print(
        f"fnorm ratio spans {min(ratios):.4f} to {max(ratios):.4f}, "
        f"a factor of {max(ratios) / min(ratios):.2f}"
    )
