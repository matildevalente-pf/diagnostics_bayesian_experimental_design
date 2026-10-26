# Tests for the interferometer system. The designed tests are the following:
# 1. Comparison of the numerical integration using the trapezoidal rule (the one in the
#    synthetic diagnostic forward model) with known analytic solutions for different
#    profiles. This isolates the quadrature.
#    1.1. Vacuum profile. Test: test_vacuum_profile_integrates_to_zero
#    1.2. Constant density profile. Test: test_constant_density_equals_density_times_length
#    1.3. Linear density profile. Test: test_linear_density_matches_analytic
#    1.4. Quadratic density profile. Test: test_quadratic_density_matches_analytic
# 2. Check that the IF signal does not depend on the orientation of the chord
#    through the plasma. Test: test_chord_reversal_invariance
#
# Run with `pytest -s` (or `-rP`) to see them:
#   pytest src/.../diagnostics_forward_models/interferometer_utils_test.py -s

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (
    interferometer_utils,
)

# Samples per chord, i.e. 59 trapezoid bins, used by every test in this file. Finer than
# the `PlasmaEstimationSettings.interferometer_n_sample = 16` the inference runs at, so
# the errors reported below are a floor on the production forward model's, not its
# actual error: the trapezoid is second order, so 16 samples cost about 15x more on a
# curved profile (and nothing at all on the constant and linear cases, which are exact).
N_SAMPLE_POINTS: int = 256

# Round-off floor for the cases the trapezoidal rule integrates exactly. Measured at
# most 4e-16 over the chords used below, so this bounds it well clear, and it is eight
# orders tighter than a quadrature-limited tolerance would have to be.
EXACT_RTOL: float = 1e-12

# Safety factor on the *predicted* quadrature error of the quadratic case, which is
# exact to leading order, so a factor of two is enough.
QUADRATIC_ERROR_MARGIN: float = 2.0

# Chords of several lengths and 3D orientations. The forward model integrates over the
# cumulative 3D arc length, so a chord that is not axis-aligned exercises that rather
# than a coordinate difference.
CHORDS: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    (np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
    (np.array([0.0, -0.5, 0.0]), np.array([0.0, 2.5, 0.0])),
    (np.array([0.0, 0.0, -3.0]), np.array([0.0, 0.0, 1.0])),
    (np.array([-1.0, -1.0, -1.0]), np.array([2.0, 0.5, 1.5])),
    (np.array([5.0, -2.0, 0.3]), np.array([-4.0, 1.0, -0.7])),
)


def _forward(
    start: np.ndarray,
    end: np.ndarray,
    density_function: Callable[[jax.Array], jax.Array],
) -> float:
    """Run the production forward line integral over a chord `start` -> `end`."""
    geometry = interferometer_utils.build_geometry_from_endpoints(
        jnp.asarray(start), jnp.asarray(end), name="test_chord"
    )
    return float(
        interferometer_utils.interferometer_forward_signal(
            geometry, density_function, N_SAMPLE_POINTS
        )
    )


def _report(case: str, trapezoid: float, analytic: float, rtol: float) -> None:
    """Print one comparison and assert it, so `pytest -s` shows every number.

    A vacuum chord has no relative error to quote, so a zero analytic value is compared
    absolutely instead, and exactly: a line integral of zeros is exactly zero.
    """
    if analytic == 0.0:
        print(f"  {case:<40} analytic=0  trapezoid={trapezoid:.6e}  (exact zero)")
        assert trapezoid == 0.0
        return
    error = abs(trapezoid - analytic) / abs(analytic)
    print(
        f"  {case:<40} analytic={analytic:.6e}  trapezoid={trapezoid:.6e}  "
        f"rel_error={error:.2e}  rtol={rtol:.1e}"
    )
    np.testing.assert_allclose(trapezoid, analytic, rtol=rtol)


def _polynomial_density(
    points: jax.Array,
    chord_start: jax.Array,
    chord_direction: jax.Array,
    length: float,
    coefficients: tuple[float, ...],
) -> jax.Array:
    """`ne(u) = sum_k coefficients[k] u^k` in the normalised arc length `u = s / L`.

    The chord's own origin, direction and length are passed in, so the profile is fixed
    in space: sampling the same chord backwards evaluates the same densities in reverse
    order rather than a mirrored profile.
    """
    arc_length = (points - chord_start) @ chord_direction
    normalised = arc_length / length
    return sum(
        coefficient * normalised**power
        for power, coefficient in enumerate(coefficients)
    )


def _polynomial_profile(
    start: np.ndarray, end: np.ndarray, coefficients: tuple[float, ...]
) -> Callable[[jax.Array], jax.Array]:
    """Bind `_polynomial_density` to one chord, ready for the forward model."""
    length = float(np.linalg.norm(end - start))
    chord_start = jnp.asarray(start)
    return functools.partial(
        _polynomial_density,
        chord_start=chord_start,
        chord_direction=(jnp.asarray(end) - chord_start) / length,
        length=length,
        coefficients=coefficients,
    )


def _analytic_polynomial(length: float, coefficients: tuple[float, ...]) -> float:
    """`L * integral_0^1 sum_k c_k u^k du = L * sum_k c_k / (k + 1)`."""
    return length * float(
        sum(coefficient / (power + 1) for power, coefficient in enumerate(coefficients))
    )


# 1.1. Vacuum: no electrons anywhere, so the line integral is exactly zero on any chord.
@pytest.mark.parametrize("start,end", CHORDS)
def test_vacuum_profile_integrates_to_zero(start: np.ndarray, end: np.ndarray) -> None:
    length = float(np.linalg.norm(end - start))
    trapezoid = _forward(start, end, _polynomial_profile(start, end, (0.0,)))
    _report(f"1.1 vacuum, L={length:.3f} m", trapezoid, 0.0, EXACT_RTOL)


# 1.2. Constant density: N = n0 * L, the trapezoidal rule being exact for a constant.
@pytest.mark.parametrize("start,end", CHORDS)
def test_constant_density_equals_density_times_length(
    start: np.ndarray, end: np.ndarray
) -> None:
    density = 3.0e19
    length = float(np.linalg.norm(end - start))
    trapezoid = _forward(start, end, _polynomial_profile(start, end, (density,)))
    _report(f"1.2 constant, L={length:.3f} m", trapezoid, density * length, EXACT_RTOL)


# 1.3. Linear density ne(u) = c0 + c1 u: N = L (c0 + c1 / 2), also exact.
@pytest.mark.parametrize("start,end", CHORDS)
@pytest.mark.parametrize("coefficients", [(1e19, 5e18), (2e19, -3e18), (0.0, 4e18)])
def test_linear_density_matches_analytic(
    start: np.ndarray, end: np.ndarray, coefficients: tuple[float, float]
) -> None:
    length = float(np.linalg.norm(end - start))
    trapezoid = _forward(start, end, _polynomial_profile(start, end, coefficients))
    _report(
        f"1.3 linear, L={length:.3f} m, c1={coefficients[1]:+.1e}",
        trapezoid,
        _analytic_polynomial(length, coefficients),
        EXACT_RTOL,
    )


# 1.4. Quadratic density ne(u) = c0 + c1 u + c2 u^2: N = L (c0 + c1/2 + c2/3), the
#      first case the trapezoidal rule does not integrate exactly.
@pytest.mark.parametrize("start,end", CHORDS)
@pytest.mark.parametrize(
    "coefficients", [(1e19, 0.0, -1e19), (5e18, 2e18, 1e18), (0.0, 0.0, 3e18)]
)
def test_quadratic_density_matches_analytic(
    start: np.ndarray, end: np.ndarray, coefficients: tuple[float, float, float]
) -> None:
    length = float(np.linalg.norm(end - start))
    trapezoid = _forward(start, end, _polynomial_profile(start, end, coefficients))
    analytic = _analytic_polynomial(length, coefficients)
    # The composite trapezoid error is `-(h^2 / 12) L f''` with `h` the sample spacing,
    # and `f'' = 2 c2 / L^2` is constant here, so the error is known exactly rather than
    # bounded, and is independent of `L`. Deriving the tolerance from it keeps the test
    # meaningful at any `N_SAMPLE_POINTS`: at 60 samples it is 1.4e-4 on a pure
    # quadratic, whatever the chord.
    spacing = length / (N_SAMPLE_POINTS - 1)
    second_derivative = 2.0 * coefficients[2] / length**2
    predicted = abs(spacing**2 * length * second_derivative / 12.0 / analytic)
    rtol = max(QUADRATIC_ERROR_MARGIN * predicted, EXACT_RTOL)
    _report(
        f"1.4 quadratic, L={length:.3f} m, c2={coefficients[2]:+.1e}",
        trapezoid,
        analytic,
        rtol,
    )


# 2. Chord reversal: dl is orientation-independent, so N agrees both ways to round-off.
#    The profile is asymmetric along the chord, so a mirrored traversal would not
#    cancel.
@pytest.mark.parametrize("start,end", CHORDS)
def test_chord_reversal_invariance(start: np.ndarray, end: np.ndarray) -> None:
    length = float(np.linalg.norm(end - start))
    profile = _polynomial_profile(start, end, (1e19, 4e18, -1.5e18))
    forward = _forward(start, end, profile)
    reversed_chord = _forward(end, start, profile)
    _report(f"2 reversal, L={length:.3f} m", reversed_chord, forward, EXACT_RTOL)
