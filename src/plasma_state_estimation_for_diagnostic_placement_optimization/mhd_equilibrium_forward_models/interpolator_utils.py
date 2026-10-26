# The equilibrium interpolator's forward pass: project a pressure profile onto
# `p(rho) = p0 (1 - rho^a)^b` and interpolate the flat target between lattice nodes.

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Int
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_utils,
    interpolator_types,
    preprocessing_types,
    preprocessing_utils,
)


def predict_y(
    lattice: interpolator_types.InterpolatorLattice,
    pressure: Float[NpOrJaxArray, " n_rho"],
) -> Float[jax.Array, " n_y"]:
    """Predict the flat target for a pressure profile sampled on the basis' grid.

    The projection onto `p(rho) = p0 (1 - rho^a)^b` is confined to the lattice's own `a`
    and `b` ranges, so a profile the lattice cannot represent is fitted at the closest
    representable shape rather than at an unreachable one.

    Args:
        lattice: Lattice to interpolate over.
        pressure: Pressure on `lattice.basis.pressure_rho_grid` [Pa].

    Returns:
        The interpolated flat target.
    """
    params = preprocessing_utils.fit_pressure_params_traceable(
        lattice.basis.pressure_rho_grid,
        pressure,
        a_range=(float(lattice.a_grid[0]), float(lattice.a_grid[-1])),
        b_range=(float(lattice.b_grid[0]), float(lattice.b_grid[-1])),
    )
    return predict_y_from_params(lattice, params)


def predict_y_from_params(
    lattice: interpolator_types.InterpolatorLattice,
    params: preprocessing_types.PressureParams,
) -> Float[jax.Array, " n_y"]:
    """Predict the flat target directly from the three pressure parameters.

    Queries outside the lattice box saturate to its boundary node without warning, because
    `bracket` clips the interpolation weight into `[0, 1]`. Use
    `fraction_outside_lattice` to find out how often that happens on a given set of
    profiles.

    Args:
        lattice: Lattice to interpolate over.
        params: Query point in the pressure parameters.

    Returns:
        The interpolated flat target.
    """
    return trilinear_interpolation(
        lattice.p0_grid,
        lattice.a_grid,
        lattice.b_grid,
        lattice.y_lattice,
        params.p0,
        params.a,
        params.b,
    )


def trilinear_interpolation(
    p0_grid: Float[NpOrJaxArray, " n_p0"],
    a_grid: Float[NpOrJaxArray, " n_a"],
    b_grid: Float[NpOrJaxArray, " n_b"],
    values: Float[NpOrJaxArray, "n_p0 n_a n_b n_values"],
    p0: ScalarFloat,
    a: ScalarFloat,
    b: ScalarFloat,
) -> Float[jax.Array, " n_values"]:
    """Interpolate over a regular three-dimensional lattice, corner by corner.

    Args:
        p0_grid: Increasing node coordinates along the first axis.
        a_grid: Increasing node coordinates along the second axis.
        b_grid: Increasing node coordinates along the third axis.
        values: Values at every node.
        p0: Query coordinate along the first axis.
        a: Query coordinate along the second axis.
        b: Query coordinate along the third axis.

    Returns:
        The weighted sum of the eight surrounding nodes.
    """
    low_0, high_0, weight_0 = bracket(p0_grid, p0)
    low_1, high_1, weight_1 = bracket(a_grid, a)
    low_2, high_2, weight_2 = bracket(b_grid, b)
    values = jnp.asarray(values)

    return (
        (1 - weight_0) * (1 - weight_1) * (1 - weight_2) * values[low_0, low_1, low_2]
        + (1 - weight_0) * (1 - weight_1) * weight_2 * values[low_0, low_1, high_2]
        + (1 - weight_0) * weight_1 * (1 - weight_2) * values[low_0, high_1, low_2]
        + (1 - weight_0) * weight_1 * weight_2 * values[low_0, high_1, high_2]
        + weight_0 * (1 - weight_1) * (1 - weight_2) * values[high_0, low_1, low_2]
        + weight_0 * (1 - weight_1) * weight_2 * values[high_0, low_1, high_2]
        + weight_0 * weight_1 * (1 - weight_2) * values[high_0, high_1, low_2]
        + weight_0 * weight_1 * weight_2 * values[high_0, high_1, high_2]
    )


def bracket(
    grid: Float[NpOrJaxArray, " n_nodes"],
    value: ScalarFloat,
) -> tuple[Int[jax.Array, ""], Int[jax.Array, ""], Float[jax.Array, ""]]:
    """Locate a value in an increasing grid, as a cell plus a position within it.

    The cell index is found with `sum(grid < value)` rather than `jnp.searchsorted`, which
    emits a `lax.scan` that `jax2onnx` cannot lower. The returned weight is clipped into
    `[0, 1]`, so a query outside the grid saturates to its nearest node instead of
    extrapolating.

    Args:
        grid: Increasing node coordinates.
        value: Query coordinate.

    Returns:
        The lower and upper node indices, and the position of `value` between them.
    """
    grid = jnp.asarray(grid)
    n_nodes = grid.shape[0]
    low = jnp.clip(jnp.sum(grid < value).astype(jnp.int32) - 1, 0, n_nodes - 2)
    high = low + 1
    span = grid[high] - grid[low]
    return (
        low,
        high,
        jnp.clip(
            jnp.where(jnp.abs(span) > 1e-30, (value - grid[low]) / span, 0.0), 0.0, 1.0
        ),
    )


def load_lattice(
    parquet_path: pathlib.Path,
    settings_path: pathlib.Path,
) -> interpolator_types.InterpolatorLattice:
    """Read a solved lattice campaign off disk and place its rows into the grid.

    The settings file lists one node per parquet row, each carrying its position in the
    lattice. The vacuum row is the exception: it has no shape indices, because a profile
    with `p0 = 0` is identically zero whatever `a` and `b` are, so the single vacuum solve
    fills all of the `p0 = 0` face rather than being repeated nine times.

    Args:
        parquet_path: Path of the lattice parquet.
        settings_path: Path of its sidecar settings JSON.

    Returns:
        The lattice, with every slot filled.
    """
    settings = json.loads(settings_path.read_text())
    metadata = dataset_utils.load_metadata(settings_path)
    dataset = dataset_utils.load_dataset(parquet_path, metadata)

    shape = (
        len(settings["p0_grid"]),
        len(settings["a_grid"]),
        len(settings["b_grid"]),
    )
    y_lattice = np.full((*shape, metadata.basis.n_y), np.nan)
    y = np.asarray(dataset.y)
    for row, node in enumerate(settings["nodes"]):
        if node["desc_equilibrium_id"] != dataset.equilibrium_ids[row]:
            raise ValueError(
                f"Node {row} of {settings_path} is {node['desc_equilibrium_id']} but row "
                f"{row} of the parquet is {dataset.equilibrium_ids[row]}."
            )
        if node["index_a"] is None:
            y_lattice[node["index_p0"], :, :] = y[row]
        else:
            y_lattice[node["index_p0"], node["index_a"], node["index_b"]] = y[row]
    if np.any(np.isnan(y_lattice)):
        empty = np.argwhere(np.isnan(y_lattice).any(axis=-1))
        raise ValueError(
            f"{settings_path} leaves {len(empty)} lattice slots unsolved, the first at "
            f"{tuple(empty[0])}."
        )

    return interpolator_types.InterpolatorLattice(
        basis=metadata.basis,
        p0_grid=np.asarray(settings["p0_grid"], dtype=float),
        a_grid=np.asarray(settings["a_grid"], dtype=float),
        b_grid=np.asarray(settings["b_grid"], dtype=float),
        y_lattice=y_lattice,
    )


def fraction_outside_lattice(
    lattice: interpolator_types.InterpolatorLattice,
    pressure: Float[NpOrJaxArray, "n_samples n_rho"],
) -> float:
    """Report how many of a set of pressure profiles fall outside the lattice box.

    The fit is deliberately run unconstrained here, unlike in `predict_y`, which confines
    it to the lattice's ranges: a query is only recognisable as outside the box if the fit
    was free to land there. A profile outside the box is still answered, by saturating to
    the boundary node, so this number is the one caveat a comparison against the networks
    has to carry.

    Args:
        lattice: Lattice the profiles are queried against.
        pressure: Pressure profiles on `lattice.basis.pressure_rho_grid` [Pa].

    Returns:
        The fraction of profiles whose fitted `(p0, a, b)` lies outside the lattice box.
    """
    params = jax.vmap(
        preprocessing_utils.fit_pressure_params_traceable, in_axes=(None, 0)
    )(jnp.asarray(lattice.basis.pressure_rho_grid), jnp.asarray(pressure))
    outside = np.zeros(np.asarray(params.p0).shape, dtype=bool)
    for values, grid in (
        (params.p0, lattice.p0_grid),
        (params.a, lattice.a_grid),
        (params.b, lattice.b_grid),
    ):
        values = np.asarray(values)
        outside |= (values < np.asarray(grid)[0]) | (values > np.asarray(grid)[-1])
    return float(np.mean(outside))
