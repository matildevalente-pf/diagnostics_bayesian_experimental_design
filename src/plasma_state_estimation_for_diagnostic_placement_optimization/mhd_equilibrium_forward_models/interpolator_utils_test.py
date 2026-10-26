import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    interpolator_types,
    interpolator_utils,
    preprocessing_types,
    testing_utils,
)

P0_GRID = np.array([0.0, 11667.0, 23333.0, 35000.0])
A_GRID = np.array([2.0, 4.0, 6.0])
B_GRID = np.array([0.5, 2.25, 4.0])


def _linear_target(p0: float, a: float, b: float, n_y: int) -> np.ndarray:
    """Evaluate a target vector that is trilinear in the pressure parameters.

    Args:
        p0: On-axis pressure of the node.
        a: Width exponent of the node.
        b: Steepness exponent of the node.
        n_y: Width of the vector.

    Returns:
        A vector that trilinear interpolation must reproduce exactly.
    """
    return np.arange(n_y, dtype=float) * (1.0 + p0 / 1.0e4) + 2.0 * a - 3.0 * b


def _linear_lattice() -> interpolator_types.InterpolatorLattice:
    """Build a lattice whose target is exactly linear in each parameter.

    Returns:
        A lattice on the real `(p0, a, b)` grids of the interpolator dataset.
    """
    basis = testing_utils.tiny_basis()
    y_lattice = np.empty((4, 3, 3, basis.n_y))
    for index_p0, index_a, index_b in itertools.product(range(4), range(3), range(3)):
        y_lattice[index_p0, index_a, index_b] = _linear_target(
            P0_GRID[index_p0], A_GRID[index_a], B_GRID[index_b], basis.n_y
        )
    return interpolator_types.InterpolatorLattice(
        basis=basis,
        p0_grid=P0_GRID,
        a_grid=A_GRID,
        b_grid=B_GRID,
        y_lattice=y_lattice,
    )


def test_the_interpolator_reproduces_every_lattice_node_exactly() -> None:
    """A query sitting on a node must return that node."""
    lattice = _linear_lattice()

    for index_p0, index_a, index_b in itertools.product(range(4), range(3), range(3)):
        predicted = interpolator_utils.predict_y_from_params(
            lattice,
            preprocessing_types.PressureParams(
                p0=P0_GRID[index_p0], a=A_GRID[index_a], b=B_GRID[index_b]
            ),
        )
        np.testing.assert_allclose(
            predicted,
            np.asarray(lattice.y_lattice)[index_p0, index_a, index_b],
            rtol=1e-12,
        )


def test_a_query_outside_the_box_saturates_to_the_boundary_node() -> None:
    """The clipped weight means an out-of-range query returns an in-range answer."""
    lattice = _linear_lattice()

    beyond = interpolator_utils.predict_y_from_params(
        lattice, preprocessing_types.PressureParams(p0=1.0e6, a=99.0, b=99.0)
    )

    np.testing.assert_allclose(
        beyond, np.asarray(lattice.y_lattice)[-1, -1, -1], rtol=1e-12
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, (0, 1, 0.0)), (0.0, (0, 1, 0.0)), (1.5, (1, 2, 0.5)), (9.0, (1, 2, 1.0))],
    ids=["below", "first_node", "midpoint", "above"],
)
def test_bracket_locates_a_value_and_clips_outside_the_grid(
    value: float, expected: tuple[int, int, float]
) -> None:
    """Below and above the grid the weight saturates rather than extrapolating."""
    low, high, weight = interpolator_utils.bracket(jnp.asarray([0.0, 1.0, 2.0]), value)

    assert (int(low), int(high), float(weight)) == expected
