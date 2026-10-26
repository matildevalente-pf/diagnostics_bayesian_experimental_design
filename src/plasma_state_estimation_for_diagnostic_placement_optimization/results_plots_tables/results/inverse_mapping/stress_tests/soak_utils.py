# Point generators and round-trip assertions of the inverse-map soak campaign:
# points are generated with an inside/outside status known by construction, inverted
# with `desc_utils.xyz_to_flux_coordinates`, forward-mapped back with
# `desc_utils.evaluate_rz`, and judged by the real-space distance to the original
# point.

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import numpy as np
from jaxtyping import Bool, Float, Int

from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)

GENERATOR_MEASUREMENT_CHORD = 0
"""Along straight chords between two boundary points at fixed `zeta` (Thomson-
scattering-like lines of sight), endpoints on the boundary included. On non-convex
cross-sections chord samples can leave the plasma; their true status comes from a
point-in-polygon test against the boundary."""

POINTS_PER_CHORD = 50
"""Evenly spaced points per measurement chord, both edge points included."""

SOLVE_TOLERANCE_FACTOR = 1e-2
"""How much tighter a shard solves than the acceptance tolerance it is judged against.

`xyz_to_flux_coordinates` freezes a point as soon as its residual drops below the tolerance
it is given, so solving at the acceptance tolerance parks a tail of points immediately below
the very threshold they are then compared with (measured: 353 of 5000 land within a factor
of two of it, and the worst reaches 0.997 `tol`). Two decades of headroom puts the worst
point of a batch at `1e-14` against a `1e-12` criterion, so no verdict is decided by the
last bits of the arithmetic. It costs nothing, since freezing masks the update rather than
skipping the work, and it keeps the solver's own non-convergence warning meaningful, which
`tol = 0` would not."""

FIT_RESOLUTION = (6, 6, 6)
"""`(L, M, N)` the test equilibrium is truncated to (via `change_resolution` in the
launcher) before the campaign runs -- low enough to keep each inversion (Jacobian
included) affordable. Recorded in the soak campaign metadata."""

_MIN_CHORD_ANGLE = 0.2
"""Minimum poloidal angle (rad) subtended by a chord: keeps interior samples much
farther from the boundary than the polygon discretization error, so the
point-in-polygon ground truth is unambiguous."""

_BOUNDARY_POLYGON_POINTS = 2048
"""Vertices of the per-`zeta` boundary polygon used to classify chord samples; the
sagitta error at this resolution is far below the closest interior sample distance
guaranteed by `_MIN_CHORD_ANGLE`."""

GENERATOR_NEAR_AXIS = 1
"""Within a tenth of the minor radius of the magnetic axis (degenerate Jacobian)."""

GENERATOR_STRADDLING_BOUNDARY = 2
"""Between 1 and 2 convergence tolerances inside or outside the boundary."""

GENERATOR_RANDOM_INTERIOR = 3
"""Uniformly random points strictly inside the boundary. The unstructured baseline: the
other three generators all sit on a feature (a chord, the axis, the boundary), so on
their own they never ask whether the map works on an ordinary interior point."""

GENERATOR_LABELS = {
    GENERATOR_MEASUREMENT_CHORD: "measurement_chord",
    GENERATOR_NEAR_AXIS: "near_axis",
    GENERATOR_STRADDLING_BOUNDARY: "straddling_boundary",
    GENERATOR_RANDOM_INTERIOR: "random_interior",
}
"""Readable label per generator, written into the campaign table so a reader of the CSV
does not need these constants."""


class InverseMapTestPoints(NamedTuple):
    """A batch of test points with inside/outside status known by construction."""

    points_xyz: Float[np.ndarray, "n_point 3"]
    generator_type: Int[np.ndarray, " n_point"]
    is_outside: Bool[np.ndarray, " n_point"]


def make_jitted_inverse_map(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    tol: float,
    maxiter: int,
    allow_extrapolation: bool = True,
) -> Callable[[Float[NpOrJaxArray, "n_point 3"]], desc_utils.FluxCoordinates]:
    """Compile the inverse map once for fixed coefficients, tolerance and batch shape.

    `xyz_to_flux_coordinates` traces its Newton `scan` on every eager call; wrapping it
    in `jax.jit` amortizes that over the many equal-shaped batches of a stress run.
    """
    return jax.jit(
        functools.partial(
            _invert,
            coefficients=coefficients,
            tol=tol,
            maxiter=maxiter,
            allow_extrapolation=allow_extrapolation,
        )
    )


def _invert(
    points_xyz: Float[NpOrJaxArray, "n_point 3"],
    coefficients: desc_types.DescFourierZernikeCoefficients,
    tol: float,
    maxiter: int,
    allow_extrapolation: bool,
) -> desc_utils.FluxCoordinates:
    return desc_utils.xyz_to_flux_coordinates(
        coefficients,
        points_xyz,
        tol=tol,
        maxiter=maxiter,
        allow_extrapolation=allow_extrapolation,
    )


def sample_inverse_map_test_points(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rng: np.random.Generator,
    n_points: int,
    boundary_offset: float,
) -> InverseMapTestPoints:
    """Draw a mixed batch of inverse-map test points.

    The batch is split into fixed fractions (so JAX sees static shapes and compiles
    once): a fifth along measurement chords, a fifth near the magnetic axis, a fifth
    uniformly random inside the boundary, and two fifths straddling the boundary -- a
    deliberate double weight on the boundary generator, where inside/outside
    classification is hardest.

    Args:
        coefficients: DESC Fourier-Zernike geometry coefficients.
        rng: Seeded generator; a batch is reproducible from its seed.
        n_points: Total number of points in the batch.
        boundary_offset: Distance scale of the boundary-straddling generator; points
            land between 1x and 2x this distance from the boundary, so it must be at
            least the inverse map's convergence tolerance for the inside/outside
            classification to be well defined.

    Returns:
        The points with their generator type and inside/outside status.
    """
    n_boundary = 2 * n_points // 5
    n_axis = n_points // 5
    n_interior = n_points // 5
    n_chord = n_points - n_boundary - n_axis - n_interior

    chord_xyz, chord_is_outside = _sample_measurement_chords(
        coefficients, rng, n_chord
    )
    axis_xyz = _sample_near_axis(coefficients, rng, n_axis)
    interior_xyz = _sample_random_interior(coefficients, rng, n_interior)
    boundary_xyz, boundary_is_outside = _sample_straddling_boundary(
        coefficients, rng, n_boundary, boundary_offset
    )

    return InverseMapTestPoints(
        points_xyz=np.concatenate([chord_xyz, axis_xyz, interior_xyz, boundary_xyz]),
        generator_type=np.concatenate(
            [
                np.full(n_chord, GENERATOR_MEASUREMENT_CHORD),
                np.full(n_axis, GENERATOR_NEAR_AXIS),
                np.full(n_interior, GENERATOR_RANDOM_INTERIOR),
                np.full(n_boundary, GENERATOR_STRADDLING_BOUNDARY),
            ]
        ),
        is_outside=np.concatenate(
            [
                chord_is_outside,
                np.zeros(n_axis + n_interior, dtype=bool),
                boundary_is_outside,
            ]
        ),
    )


def inverse_map_failure_mask(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    test_points: InverseMapTestPoints,
    flux: desc_utils.FluxCoordinates,
    tol: float,
    allow_extrapolation: bool = False,
) -> tuple[Bool[np.ndarray, " n_point"], Float[np.ndarray, " n_point"]]:
    """Per-point acceptance verdict and real-space round-trip accuracy.

    Returns `(failing, achieved_accuracy)`: `failing[i]` is True when point `i`
    violates the criterion (inside points must round-trip below `tol`; outside points
    must fail to converge without extrapolation, or converge to `rho > 1` with it).
    `rho` is the true final iterate (never NaN), so convergence is read off the
    round-trip distance. The soak campaign records the failing points.

    The side-of-the-boundary requirement is only applied where it is decidable: a point
    solved to within `tol` of the last closed flux surface (`boundary_distance`) is inside
    the map's own accuracy of it, so calling one side right and the other wrong would be
    reading round-off rather than testing the solver.

    `tol` here is the acceptance criterion, which the caller is expected to solve well
    inside of (see `SOLVE_TOLERANCE_FACTOR`); judging a point that was solved only to `tol`
    compares two numbers that agree to a couple of ulp against a threshold they both sit
    on.
    """
    points = test_points.points_xyz
    rho = np.asarray(flux.rho)
    r, z = desc_utils.evaluate_rz(
        coefficients,
        rho,
        np.asarray(flux.theta),
        np.asarray(flux.zeta),
        extend_beyond_boundary=allow_extrapolation,
    )
    target_r = np.hypot(points[:, 0], points[:, 1])
    achieved_accuracy = np.hypot(np.asarray(r) - target_r, np.asarray(z) - points[:, 2])
    converged_on_target = achieved_accuracy < tol
    # A wrong-side verdict only counts where the side of the boundary is resolvable at
    # all: a point whose solved position is less than `tol` from the boundary is inside
    # the map's own accuracy, so which side it lands on is round-off, not an error.
    resolvable = boundary_distance(coefficients, flux, allow_extrapolation) > tol
    outside_ok = (
        converged_on_target & ((rho > 1.0) | ~resolvable)
        if allow_extrapolation
        else ~converged_on_target | ~resolvable
    )
    point_ok = np.where(test_points.is_outside, outside_ok, converged_on_target)
    return ~point_ok, achieved_accuracy


def boundary_distance(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    flux: desc_utils.FluxCoordinates,
    allow_extrapolation: bool = False,
) -> Float[np.ndarray, " n_point"]:
    """Real-space distance (m) from the last closed flux surface of each solved point.

    `|rho - 1|` scaled by `|d(R, Z)/drho|`, so it is directly comparable with the
    convergence tolerance: a solved point closer to the boundary than `tol` is within the
    map's own accuracy of it, and its side of the boundary is not determined. Shared by
    the acceptance criterion and the recorded `failure_kind` so the two cannot disagree.
    """
    jacobian = desc_utils.evaluate_rz_jacobian(
        coefficients,
        np.asarray(flux.rho),
        np.asarray(flux.theta),
        np.asarray(flux.zeta),
        extend_beyond_boundary=allow_extrapolation,
    )
    radial_scale = np.hypot(
        np.asarray(jacobian.dr_drho), np.asarray(jacobian.dz_drho)
    )
    return np.abs(np.asarray(flux.rho) - 1.0) * radial_scale


def _sample_measurement_chords(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rng: np.random.Generator,
    n_points: int,
) -> tuple[Float[np.ndarray, "n_point 3"], Bool[np.ndarray, " n_point"]]:
    """Points along straight boundary-to-boundary chords at fixed `zeta`, mimicking the
    measurement positions of a Thomson-scattering laser line of sight.

    Each chord connects two random boundary points in a `zeta = const` plane and
    carries `POINTS_PER_CHORD` evenly spaced points, the two boundary endpoints
    included -- so every chord also exercises the `rho = 1` edge exactly. On
    non-convex cross-sections (bean shapes) a chord can leave the plasma, so each
    interior sample is classified by a point-in-polygon test against a dense boundary
    polygon of its own `zeta` plane -- ground truth independent of the solver under
    test. The chord endpoints subtend at least `_MIN_CHORD_ANGLE` so interior samples
    stay far from the boundary relative to the polygon discretization error. The last
    chord is truncated so exactly `n_points` points are returned.
    """
    n_chords = -(-n_points // POINTS_PER_CHORD)
    zeta = rng.uniform(0.0, 2.0 * np.pi, n_chords)
    theta_start = rng.uniform(0.0, 2.0 * np.pi, n_chords)
    theta_end = theta_start + rng.uniform(
        _MIN_CHORD_ANGLE, 2.0 * np.pi - _MIN_CHORD_ANGLE, n_chords
    )
    ones = np.ones(n_chords)

    r_start, z_start = desc_utils.evaluate_rz(coefficients, ones, theta_start, zeta)
    r_end, z_end = desc_utils.evaluate_rz(coefficients, ones, theta_end, zeta)
    r_start, z_start = np.asarray(r_start), np.asarray(z_start)
    r_end, z_end = np.asarray(r_end), np.asarray(z_end)

    fraction = np.linspace(0.0, 1.0, POINTS_PER_CHORD)[None, :]
    r = r_start[:, None] + fraction * (r_end - r_start)[:, None]
    z = z_start[:, None] + fraction * (z_end - z_start)[:, None]

    theta_polygon = np.linspace(0.0, 2.0 * np.pi, _BOUNDARY_POLYGON_POINTS, False)
    r_polygon, z_polygon = desc_utils.evaluate_rz(
        coefficients,
        np.ones(n_chords * _BOUNDARY_POLYGON_POINTS),
        np.tile(theta_polygon, n_chords),
        np.repeat(zeta, _BOUNDARY_POLYGON_POINTS),
    )
    shape = (n_chords, _BOUNDARY_POLYGON_POINTS)
    inside = _points_inside_polygon(
        r, z, np.asarray(r_polygon).reshape(shape), np.asarray(z_polygon).reshape(shape)
    )
    is_outside = ~inside
    # The endpoints sit exactly on the boundary, where the polygon test is
    # ill-conditioned; they are inside by construction (rho = 1).
    is_outside[:, 0] = False
    is_outside[:, -1] = False

    zeta_points = np.repeat(zeta, POINTS_PER_CHORD)
    return (
        _cylindrical_to_xyz(r.ravel(), zeta_points, z.ravel())[:n_points],
        is_outside.ravel()[:n_points],
    )


def _points_inside_polygon(
    r: Float[np.ndarray, "n_chord n_sample"],
    z: Float[np.ndarray, "n_chord n_sample"],
    polygon_r: Float[np.ndarray, "n_chord n_vertex"],
    polygon_z: Float[np.ndarray, "n_chord n_vertex"],
) -> Bool[np.ndarray, "n_chord n_sample"]:
    """Even-odd (ray casting) point-in-polygon test, one closed polygon per chord."""
    r_a, z_a = polygon_r[:, None, :], polygon_z[:, None, :]
    r_b = np.roll(polygon_r, -1, axis=1)[:, None, :]
    z_b = np.roll(polygon_z, -1, axis=1)[:, None, :]
    z_point = z[:, :, None]
    crosses = (z_a > z_point) != (z_b > z_point)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_intersect = r_a + (z_point - z_a) * (r_b - r_a) / (z_b - z_a)
    hits = crosses & (r[:, :, None] < r_intersect)
    return (hits.sum(axis=2) % 2) == 1


def _sample_near_axis(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rng: np.random.Generator,
    n_points: int,
) -> Float[np.ndarray, "n_point 3"]:
    """Magnetic axis at a random toroidal angle, offset by up to a tenth of the minor
    radius -- the hard case, where the `(rho, theta)` Jacobian degenerates."""
    zeta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    zeros = np.zeros(n_points)
    r_axis, z_axis = desc_utils.evaluate_rz(coefficients, zeros, zeros, zeta)

    offset_scale = _approximate_minor_radius(coefficients) / 10.0
    dr = offset_scale * rng.uniform(-1.0, 1.0, n_points)
    dz = offset_scale * rng.uniform(-1.0, 1.0, n_points)
    return _cylindrical_to_xyz(np.asarray(r_axis) + dr, zeta, np.asarray(z_axis) + dz)


def _sample_random_interior(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rng: np.random.Generator,
    n_points: int,
) -> Float[np.ndarray, "n_point 3"]:
    """Uniformly random points strictly inside the boundary, drawn in flux coordinates
    and forward-mapped -- inside by construction, no rejection step, static shape.

    `rho = sqrt(u)` with `u` uniform spreads the points through the volume instead of
    piling them near the axis, and `theta` and `zeta` are uniform. Unlike the other
    three generators this one is unstructured, so it is the baseline that says the map
    works on an ordinary interior point and not only on the hard cases.
    """
    rho = np.sqrt(rng.uniform(0.0, 1.0, n_points))
    theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    zeta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    r, z = desc_utils.evaluate_rz(coefficients, rho, theta, zeta)
    return _cylindrical_to_xyz(np.asarray(r), zeta, np.asarray(z))


def _sample_straddling_boundary(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    rng: np.random.Generator,
    n_points: int,
    boundary_offset: float,
) -> tuple[Float[np.ndarray, "n_point 3"], Bool[np.ndarray, " n_point"]]:
    """Points on the boundary, stepped along the outward normal by a distance in
    `[boundary_offset, 2 * boundary_offset]` -- 50/50 inward or outward.

    The step is at least `boundary_offset` so the inverse map can reliably decide
    which side of the boundary a point is on. The normal is built from the poloidal
    tangent `(dR/dtheta, dZ/dtheta)` and oriented outward against the axis-to-boundary
    direction, which is robust to the poloidal-angle handedness of the equilibrium.
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    zeta = rng.uniform(0.0, 2.0 * np.pi, n_points)
    ones = np.ones(n_points)
    zeros = np.zeros(n_points)

    r_boundary, z_boundary = desc_utils.evaluate_rz(coefficients, ones, theta, zeta)
    r_boundary, z_boundary = np.asarray(r_boundary), np.asarray(z_boundary)
    jacobian = desc_utils.evaluate_rz_jacobian(coefficients, ones, theta, zeta)
    dr_dtheta = np.asarray(jacobian.dr_dtheta)
    dz_dtheta = np.asarray(jacobian.dz_dtheta)
    tangent_norm = np.hypot(dr_dtheta, dz_dtheta)
    normal_r = dz_dtheta / tangent_norm
    normal_z = -dr_dtheta / tangent_norm

    r_axis, z_axis = desc_utils.evaluate_rz(coefficients, zeros, zeros, zeta)
    outward_sign = np.sign(
        normal_r * (r_boundary - np.asarray(r_axis))
        + normal_z * (z_boundary - np.asarray(z_axis))
    )
    normal_r *= outward_sign
    normal_z *= outward_sign

    # `evaluate_rz` carries an absolute roundoff of order `eps * R`; when
    # `boundary_offset` approaches that scale (e.g. tol = 1e-12 on a 10 m machine) the
    # nominal offset alone no longer guarantees which side of the tolerance the true
    # distance lands on, so add it as an explicit safety margin.
    roundoff = 64.0 * np.finfo(float).eps * np.max(np.hypot(r_boundary, z_boundary))
    is_outside = rng.uniform(size=n_points) < 0.5
    distance = boundary_offset * (1.0 + rng.uniform(size=n_points)) + roundoff
    step = np.where(is_outside, distance, -distance)
    return (
        _cylindrical_to_xyz(
            r_boundary + step * normal_r, zeta, z_boundary + step * normal_z
        ),
        is_outside,
    )


def _approximate_minor_radius(
    coefficients: desc_types.DescFourierZernikeCoefficients,
) -> float:
    """Mean axis-to-boundary distance over a coarse `(theta, zeta)` grid."""
    theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    zeta = np.linspace(
        0.0, 2.0 * np.pi / coefficients.n_field_periods, 8, endpoint=False
    )
    theta_grid, zeta_grid = (a.ravel() for a in np.meshgrid(theta, zeta))
    r_boundary, z_boundary = desc_utils.evaluate_rz(
        coefficients, np.ones_like(theta_grid), theta_grid, zeta_grid
    )
    r_axis, z_axis = desc_utils.evaluate_rz(
        coefficients, np.zeros_like(theta_grid), np.zeros_like(theta_grid), zeta_grid
    )
    return float(
        np.mean(
            np.hypot(
                np.asarray(r_boundary) - np.asarray(r_axis),
                np.asarray(z_boundary) - np.asarray(z_axis),
            )
        )
    )


def _cylindrical_to_xyz(
    r: Float[np.ndarray, " n_point"],
    phi: Float[np.ndarray, " n_point"],
    z: Float[np.ndarray, " n_point"],
) -> Float[np.ndarray, "n_point 3"]:
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
