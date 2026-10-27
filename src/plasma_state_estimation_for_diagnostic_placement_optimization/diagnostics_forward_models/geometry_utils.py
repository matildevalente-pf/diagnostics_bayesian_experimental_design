# Poloidal-cross-section geometry placing the laser, the chords and the optics of a
# layout on one surface. Closed curves are `(2, n)` arrays of `(R, Z)` in metres.

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import scipy.interpolate
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (
    geometry_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (
    interferometer_types as if_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (
    interferometer_utils as if_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (
    thomson_scattering_types as ts_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (
    thomson_scattering_utils as ts_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)

DEFAULT_SURFACE_OFFSET: float = 0.14
"""Distance in m of the auxiliary design surface outside the plasma boundary."""

THETA_MIN_DEG: float = 10.0
THETA_MAX_DEG: float = 170.0
"""Hard bounds on the scattering angle."""

LOW_ANGLE_RANGE_DEG: tuple[float, float] = (10.0, 50.0)
"""Scattering-angle range of the forward-scatter optic set, which views the high-field
side (volumes at a major radius below the magnetic axis)."""

HIGH_ANGLE_RANGE_DEG: tuple[float, float] = (130.0, 170.0)
"""Scattering-angle range of the back-scatter optic set, which views the low-field side
(volumes at a major radius above the magnetic axis)."""

"""Floor of the volume spacing density, keeping it strictly positive."""

_FLUX_CACHE: dict = {}
_SURFACE_CACHE: dict = {}


def boundary_cross_section(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: ScalarFloat,
    n_theta: int = 360,
) -> jt.Float[jax.Array, "2 n_theta"]:
    """Closed `(R, Z)` polygon of the last closed flux surface at one toroidal angle."""
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_theta + 1)[:-1]
    r, z = desc_utils.evaluate_rz(
        coefficients, jnp.ones_like(theta), theta, jnp.full_like(theta, toroidal_angle)
    )
    return jnp.stack([r, z])


def magnetic_axis(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: ScalarFloat,
) -> jt.Float[np.ndarray, " 2"]:
    """`(R, Z)` of the magnetic axis (`rho = 0`) at one toroidal angle."""
    # Evaluated eagerly even inside a trace, since the plane is fixed data.
    with jax.ensure_compile_time_eval():
        r, z = desc_utils.evaluate_rz(
            coefficients, jnp.zeros(1), jnp.zeros(1), jnp.array([toroidal_angle])
        )
        return np.array([float(r[0]), float(z[0])])


def cross_section_and_surface(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: ScalarFloat,
    surface_offset: ScalarFloat,
) -> tuple[jt.Float[jax.Array, "2 n_theta"], geometry_types.PoloidalSurface]:
    """Plasma polygon and the auxiliary design surface a layout is placed on, cached.

    Only the equilibrium, the toroidal angle and the surface offset change these, so an
    interactive layout that moves the chord or the optics does not pay for them again.
    """
    key = (
        id(coefficients),
        round(float(toroidal_angle), 6),
        round(float(surface_offset), 6),
    )
    if key not in _SURFACE_CACHE:
        # Evaluated eagerly even inside a trace, since the plane is fixed data.
        with jax.ensure_compile_time_eval():
            plasma = boundary_cross_section(coefficients, toroidal_angle)
        curve = design_surface_curve(plasma, surface_offset)
        _SURFACE_CACHE[key] = (
            plasma,
            geometry_types.PoloidalSurface(
                curve=curve,
                centre=curve.mean(axis=1),
                toroidal_angle=float(toroidal_angle),
            ),
        )
    return _SURFACE_CACHE[key]


def design_surface_curve(
    curve: jt.Float[NpOrJaxArray, "2 n_point"],
    offset: ScalarFloat,
    n_ray: int = 360,
    n_bisect: int = 40,
) -> jt.Float[np.ndarray, "2 n_ray"]:
    """Surface at constant distance `offset` outside a closed curve."""
    curve = np.asarray(curve)
    centre = curve.mean(axis=1)
    angle = np.linspace(-np.pi, np.pi, n_ray, endpoint=False)
    ray = np.stack([np.cos(angle), np.sin(angle)])
    inner = np.zeros(n_ray)
    outer = np.full(
        n_ray,
        np.max(np.linalg.norm(curve - centre[:, None], axis=0))
        + 2.0 * float(offset)
        + 1e-9,
    )
    for _ in range(n_bisect):
        middle = 0.5 * (inner + outer)
        closer = signed_distance(curve, centre[:, None] + middle * ray) < offset
        inner = np.where(closer, middle, inner)
        outer = np.where(closer, outer, middle)
    return centre[:, None] + (0.5 * (inner + outer)) * ray


# Built by *forward* mapping a `(rho, theta)` grid rather than by the Newton inverse
# map, which costs ~58 ms per point and is far too slow to drive an interactive layout;
# this is built once per toroidal angle and queried in microseconds for ~4e-4 accuracy
# in `rho`. Use `inverse_mapping/` when accuracy matters more than latency.
def flux_interpolator(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: ScalarFloat,
    n_rho: int = 48,
    n_theta: int = 145,
) -> scipy.interpolate.LinearNDInterpolator:
    """`(R, Z) -> rho` in one poloidal plane, cached per toroidal angle.

    Returns `nan` outside the triangulated region. Because the triangulation's hull is
    convex, a concave cross-section leaves points that are outside the plasma yet still
    interpolated, so gate on `points_inside` rather than on `nan`.
    """
    key = (id(coefficients), round(float(toroidal_angle), 6), n_rho, n_theta)
    if key not in _FLUX_CACHE:
        rho_grid, theta_grid = np.meshgrid(
            np.linspace(0.0, 1.0, n_rho),
            np.linspace(0.0, 2.0 * np.pi, n_theta),
            indexing="ij",
        )
        r, z = desc_utils.evaluate_rz(
            coefficients,
            jnp.asarray(rho_grid.ravel()),
            jnp.asarray(theta_grid.ravel()),
            jnp.full(rho_grid.size, toroidal_angle),
        )
        _FLUX_CACHE[key] = scipy.interpolate.LinearNDInterpolator(
            np.stack([np.asarray(r), np.asarray(z)], axis=1), rho_grid.ravel()
        )
    return _FLUX_CACHE[key]


def points_inside(
    curve: jt.Float[NpOrJaxArray, "2 n_point"],
    points: jt.Float[NpOrJaxArray, "2 n_query"],
) -> jt.Bool[np.ndarray, " n_query"]:
    """Even-odd ray cast: which `points` lie inside the closed `curve`.

    Makes no convexity or star-shapedness assumption, so it is the reliable inside test
    for a bean-shaped cross-section.
    """
    x, y = np.asarray(points)
    cx, cy = np.asarray(curve)
    nx, ny = np.roll(cx, -1), np.roll(cy, -1)
    straddles = ((cy[:, None] <= y) & (ny[:, None] > y)) | (
        (ny[:, None] <= y) & (cy[:, None] > y)
    )
    denominator = np.where(ny == cy, 1.0, ny - cy)[:, None]
    crossing_x = cx[:, None] + (y - cy[:, None]) / denominator * (nx - cx)[:, None]
    return np.sum(straddles & (crossing_x > x), axis=0) % 2 == 1


def signed_distance(
    curve: jt.Float[NpOrJaxArray, "2 n_point"],
    points: jt.Float[NpOrJaxArray, "2 n_query"],
) -> jt.Float[np.ndarray, " n_query"]:
    """Distance from each point to the closed `curve`, negative inside it."""
    curve = np.asarray(curve)
    points = np.asarray(points)
    segment = (np.roll(curve, -1, axis=1) - curve)[:, :, None]
    delta = points[:, None, :] - curve[:, :, None]
    length_sq = np.sum(segment**2, axis=0)
    fraction = np.clip(
        np.sum(delta * segment, axis=0) / np.where(length_sq > 0, length_sq, 1.0),
        0.0,
        1.0,
    )
    closest = delta - fraction[None, :, :] * segment
    distance = np.min(np.sqrt(np.sum(closest**2, axis=0)), axis=0)
    return np.where(points_inside(curve, points), -distance, distance)


def surface_point_at_angle(
    surface: geometry_types.PoloidalSurface,
    angle: NpOrJaxArray | ScalarFloat,
) -> jt.Float[jax.Array, "2 n_angle"]:
    """Points of `surface` seen from its centre at poloidal `angle` (radians).

    The one place a layout turns an angle into a position, so the laser, the chords and
    the optic ports are placed the same way. Differentiable in `angle`, which is what
    lets the placement optimization move a line of sight.
    """
    angle = jnp.atleast_1d(jnp.asarray(angle, dtype=float))
    centre = jnp.asarray(surface.centre)
    relative = jnp.asarray(surface.curve) - centre[:, None]
    angles = jnp.arctan2(relative[1], relative[0])
    order = jnp.argsort(angles)
    radius = jnp.interp(
        angle,
        angles[order],
        jnp.linalg.norm(relative, axis=0)[order],
        period=2.0 * jnp.pi,
    )
    return centre[:, None] + radius[None, :] * jnp.stack(
        [jnp.cos(angle), jnp.sin(angle)]
    )


def sampled_span(
    span: tuple[ScalarFloat, ScalarFloat] | None,
    length: ScalarFloat,
    delta: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat]:
    """Chord fractions to place volumes over: the plasma span widened by `delta` metres.

    Sampling only here, rather than over the whole beam, is what stops most volumes from
    measuring vacuum. `delta` keeps a margin outside the boundary; `delta = 0` samples
    exactly the plasma path.
    """
    if span is None:
        return 0.5, 0.5
    delta_s = delta / length
    return jnp.maximum(0.0, span[0] - delta_s), jnp.minimum(1.0, span[1] + delta_s)


def scattering_angles(
    volumes: jt.Float[NpOrJaxArray, "2 n_volume"],
    optic: jt.Float[NpOrJaxArray, " 2"],
    direction: jt.Float[NpOrJaxArray, " 2"],
) -> jt.Float[np.ndarray, " n_volume"]:
    """Scattering angle (radians) at each volume for one optic.

    The angle between the laser propagation direction and the volume-to-optic line of
    sight: `0` is pure forward scatter, `pi` pure back scatter. This is the `theta` that
    sets the spectral width and enters ``dL ~ 1 / sin(theta)``.
    """
    sight = np.asarray(optic).reshape(2, 1) - np.asarray(volumes)
    sight = sight / np.linalg.norm(sight, axis=0, keepdims=True)
    return np.arccos(np.clip(np.asarray(direction) @ sight, -1.0, 1.0))


def admissible_surface_angles(
    served: jt.Float[NpOrJaxArray, "2 n_volume"],
    surface: geometry_types.PoloidalSurface,
    direction: jt.Float[NpOrJaxArray, " 2"],
    limits: tuple[float, float],
    object_distance: ScalarFloat,
    step_deg: float = 2.0,
) -> jt.Float[np.ndarray, " n_admissible"]:
    """Surface angles along which a lens at `object_distance` meets the `theta` limits.

    Args:
        served: `(R, Z)` of the volumes the optic set reads.
        surface: The surface the optic ports sit on.
        direction: Unit laser direction in the poloidal plane.
        limits: Admissible scattering-angle range of the set in degrees.
        object_distance: Lens-to-volume distance in m (`optic_position`).
        step_deg: Spacing of the candidate surface angles in degrees.

    Returns:
        The admissible surface angles in rad.
    """
    served = np.asarray(served)
    if served.shape[1] == 0:
        return np.zeros(0)
    candidates = np.radians(np.arange(-180.0, 180.0, step_deg))
    points = np.asarray(surface_point_at_angle(surface, candidates))
    low, high = limits
    admissible = []
    for index in range(candidates.size):
        optic = optic_position(served, points[:, index], object_distance)
        theta = np.degrees(scattering_angles(served, optic, direction))
        admissible.append(bool(theta.min() >= low and theta.max() <= high))
    return candidates[np.array(admissible, dtype=bool)]


def optic_position(
    served: jt.Float[NpOrJaxArray, "2 n_volume"],
    surface_point: jt.Float[NpOrJaxArray, " 2"],
    object_distance: ScalarFloat,
    mask: jt.Bool[NpOrJaxArray, " n_volume"] | None = None,
) -> jt.Float[jax.Array, " 2"]:
    """Lens position: `object_distance` from the served volumes' centroid, towards the
    surface point.

    The surface point fixes only the viewing direction; the standoff is the lens
    design's (`thomson_scattering_utils.collection_object_distance`), so the lens ends
    up beyond the design surface, as at W7-X, whose optics stand in immersion tubes that
    carry the vacuum barrier (Pasch 2016, section II,
    https://doi.org/10.1063/1.4962248) about 1.3 m from the laser beam (Bozhenkov 2017,
    section 2, https://doi.org/10.1088/1748-0221/12/10/P10004).

    Args:
        served: `(R, Z)` of the volumes the lens reads.
        surface_point: `(R, Z)` of the design-surface point the lens looks through.
        object_distance: Lens-to-volume distance in m.
        mask: Which of `served` the lens reads; None, or a mask selecting none of
            them, reads them all.

    Returns:
        `(R, Z)` of the lens.
    """
    served = jnp.asarray(served)
    if mask is None:
        centroid = served.mean(axis=1)
    else:
        # A weighted mean instead of `served[:, mask]`, whose shape a trace cannot fix.
        weights = jnp.where(jnp.any(mask), jnp.asarray(mask, dtype=float), 1.0)
        centroid = served @ weights / jnp.sum(weights)
    towards = jnp.asarray(surface_point) - centroid
    return centroid + object_distance * towards / jnp.linalg.norm(towards)


def uniform_volume_fractions(n_volumes: int) -> tuple[float, ...]:
    """Centres of `n_volumes` equal cells along the laser's plasma span, in `[0, 1]`."""
    return tuple(((np.arange(n_volumes) + 0.5) / n_volumes).tolist())


def chord_line_integrated_density(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: ScalarFloat,
    plasma: jt.Float[NpOrJaxArray, "2 n_point"],
    start_rz: jt.Float[NpOrJaxArray, " 2"],
    end_rz: jt.Float[NpOrJaxArray, " 2"],
    electron_density_fn: Callable[[NpOrJaxArray], NpOrJaxArray],
    n_sample: int = 128,
) -> float:
    """Line-integrated electron density along a chord lying in one poloidal plane.

    Wraps the production kernel `interferometer_utils.interferometer_forward_signal`, so
    the number here is the same quantity the inference compares against, not a
    re-derivation. The chord endpoints are given in `(R, Z)`; they are lifted to the
    lab frame at `toroidal_angle` before the integral.

    The density outside the last closed flux surface is zero, so a chord that misses the
    plasma integrates to zero rather than to a spurious value.
    """
    cos_angle, sin_angle = np.cos(toroidal_angle), np.sin(toroidal_angle)
    geometry = if_utils.build_geometry_from_endpoints(
        np.array([start_rz[0] * cos_angle, start_rz[0] * sin_angle, start_rz[1]]),
        np.array([end_rz[0] * cos_angle, end_rz[0] * sin_angle, end_rz[1]]),
        name="poloidal_chord",
    )
    density_at = functools.partial(
        _electron_density_at_points,
        plasma=plasma,
        interpolator=flux_interpolator(coefficients, toroidal_angle),
        electron_density_fn=electron_density_fn,
    )
    return float(
        if_utils.interferometer_forward_signal(geometry, density_at, n_sample=n_sample)
    )


def _electron_density_at_points(
    points_xyz: jt.Float[NpOrJaxArray, "n_point 3"],
    plasma: jt.Float[NpOrJaxArray, "2 n_point"],
    interpolator: scipy.interpolate.LinearNDInterpolator,
    electron_density_fn: Callable[[NpOrJaxArray], NpOrJaxArray],
) -> jt.Float[NpOrJaxArray, " n_point"]:
    """Electron density at lab-frame points, zero outside the last closed flux surface.

    The density field `interferometer_forward_signal` samples along a chord. Points are
    projected back into the poloidal plane, labelled with `rho` by `interpolator`, and
    handed to `electron_density_fn`. Outside points are given `rho = 1` before that call
    (the interpolant is unreliable there and the profile must still be evaluated on the
    whole batch) and then zeroed, so a chord that misses the plasma integrates to zero
    rather than to a spurious value.
    """
    points = np.asarray(points_xyz)
    radius_rz = np.hypot(points[:, 0], points[:, 1])
    height = points[:, 2]
    inside = points_inside(plasma, np.stack([radius_rz, height]))
    flux_rho = interpolator(radius_rz, height)
    flux_rho = np.where(inside & np.isfinite(flux_rho), np.nan_to_num(flux_rho), 1.0)
    return jnp.asarray(
        np.where(inside, np.asarray(electron_density_fn(jnp.asarray(flux_rho))), 0.0)
    )


def lab_frame_points(
    points_rz: jt.Float[NpOrJaxArray, "2 n_point"], toroidal_angle: ScalarFloat
) -> jt.Float[jax.Array, "n_point 3"]:
    """Lift `(R, Z)` points in one poloidal plane to lab-frame `(x, y, z)`."""
    points_rz = jnp.asarray(points_rz)
    cos_angle, sin_angle = jnp.cos(toroidal_angle), jnp.sin(toroidal_angle)
    return jnp.stack(
        [points_rz[0] * cos_angle, points_rz[0] * sin_angle, points_rz[1]], axis=-1
    )


def thomson_volumes_on_chord(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: ScalarFloat = 0.0,
    surface_offset: ScalarFloat = DEFAULT_SURFACE_OFFSET,
    laser_surface_angles: tuple[float, float] = (0.0, np.pi),
    delta: ScalarFloat = 0.0,
    volume_fractions: tuple[float, ...] = (),
) -> tuple[
    jt.Float[jax.Array, "2 n_element"],
    jt.Float[jax.Array, " 2"],
    jt.Float[jax.Array, " 2"],
]:
    """Scattering-volume centres, the laser direction, and its design-surface entry.

    The beam runs from the first of `laser_surface_angles` to the second; each volume
    sits at its fraction of the beam's plasma span widened by `delta`, 0 the entry side.

    Returns `(volumes_rz, direction, entry_rz)`, all in the poloidal plane.
    """
    plasma, surface = cross_section_and_surface(
        coefficients, toroidal_angle, surface_offset
    )
    ends = surface_point_at_angle(surface, jnp.asarray(laser_surface_angles))
    entry, exit_point = ends[:, 0], ends[:, 1]
    length = jnp.linalg.norm(exit_point - entry)
    span_low, span_high, crosses = _segment_curve_crossings(plasma, entry, exit_point)
    # A traced beam that misses the plasma cannot raise; its volumes come out infinite.
    if not isinstance(crosses, jax.core.Tracer) and not bool(crosses):  # type: ignore
        raise ValueError(
            f"the Thomson beam between surface angles "
            f"{float(laser_surface_angles[0]):.4f} "
            f"and {float(laser_surface_angles[1]):.4f} rad misses the plasma at "
            f"toroidal angle {float(toroidal_angle):.4f} rad, so no scattering volume "
            f"would carry signal"
        )
    span_low, span_high = sampled_span((span_low, span_high), length, delta)
    fractions = jnp.asarray(volume_fractions, dtype=float)
    positions = span_low + fractions * (span_high - span_low)
    volumes = entry[:, None] + positions[None, :] * (exit_point - entry)[:, None]
    return volumes, (exit_point - entry) / length, entry


def thomson_geometry_from_chord(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: ScalarFloat = 0.0,
    surface_offset: ScalarFloat = DEFAULT_SURFACE_OFFSET,
    laser_surface_angles: tuple[float, float] = (0.0, np.pi),
    delta: ScalarFloat = 0.0,
    volume_fractions: tuple[float, ...] = (),
    optic_surface_angles: tuple[float, float] = (-np.pi * 125.0 / 180.0, np.pi * 0.25),
    aperture_diameter: ScalarFloat = ts_utils.W7X_COLLECTION_APERTURE_DIAMETER,
    focal_length: ScalarFloat = ts_utils.W7X_COLLECTION_FOCAL_LENGTH,
    fiber_core_diameter: ScalarFloat = ts_utils.W7X_FIBER_BUNDLE_DIAMETER,
    magnification: ScalarFloat = ts_utils.W7X_COLLECTION_MAGNIFICATION,
    name: str = "thomson",
    aim_fractions: tuple[float, ...] | None = None,
) -> ts_types.ThomsonScatteringGeometry:
    """The geometry of a chord viewed by two lenses, one per side of the magnetic axis.

    Each lens looks through the design surface at its angle and sits at the standoff its
    design images from (`optic_position`); high-field-side volumes go to the first lens,
    low-field-side ones to the second.

    Args:
        coefficients: DESC geometry coefficients of the equilibrium.
        toroidal_angle: Poloidal plane of the layout in rad.
        surface_offset: Distance of the design surface outside the plasma, in m.
        laser_surface_angles: The beam's two surface angles, entry first, in rad.
        delta: Margin outside the plasma the volumes may spread over, in m.
        volume_fractions: Position of each volume along the beam's plasma span, 0 to 1.
        optic_surface_angles: Surface angle of the forward-scatter lens, which serves
            the high-field-side volumes, then of the back-scatter one, in rad.
        aperture_diameter: Clear aperture of the lenses in m.
        focal_length: Focal length of the lenses in m.
        fiber_core_diameter: Diameter of the fiber bundle a volume is imaged onto, in m.
        magnification: Design magnification, fixing the standoff.
        name: Label of the geometry.
        aim_fractions: Span positions whose centroid on each side of the axis a lens
            aims at; None aims each lens at the volumes it serves.

    Returns:
        The geometry, with the two lens positions in `optic_positions`.

    Raises:
        ValueError: If a volume falls outside `[THETA_MIN_DEG, THETA_MAX_DEG]`, where
            `dL ~ 1 / sin(theta)` diverges; a set outside its narrower range only warns.
    """
    toroidal_angle = float(toroidal_angle)
    _, surface = cross_section_and_surface(coefficients, toroidal_angle, surface_offset)
    axis = magnetic_axis(coefficients, toroidal_angle)
    if len(volume_fractions) == 0:
        raise ValueError("a Thomson geometry needs at least one scattering volume")
    volumes, direction, entry = thomson_volumes_on_chord(
        coefficients,
        toroidal_angle,
        surface_offset,
        laser_surface_angles,
        delta,
        volume_fractions,
    )
    is_hfs = volumes[0] < axis[0]
    object_distance = ts_utils.collection_object_distance(focal_length, magnification)
    ports = surface_point_at_angle(surface, jnp.asarray(optic_surface_angles))
    # Aiming at fixed span positions rather than at the served volumes keeps one
    # layout's lenses where they are whatever volumes another layout puts on the beam.
    aimed = (
        volumes
        if aim_fractions is None
        else thomson_volumes_on_chord(
            coefficients,
            toroidal_angle,
            surface_offset,
            laser_surface_angles,
            delta,
            aim_fractions,
        )[0]
    )
    aimed_is_hfs = aimed[0] < axis[0]
    optics_rz = jnp.stack(
        [
            optic_position(aimed, ports[:, index], object_distance, served)
            for index, served in enumerate((aimed_is_hfs, ~aimed_is_hfs))
        ],
        axis=1,
    )
    cos_angle, sin_angle = np.cos(toroidal_angle), np.sin(toroidal_angle)
    geometry = ts_types.ThomsonScatteringGeometry(
        laser_origin=lab_frame_points(entry[:, None], toroidal_angle)[0],
        laser_direction=jnp.stack(
            [direction[0] * cos_angle, direction[0] * sin_angle, direction[1]]
        ),
        reference_phi=toroidal_angle,
        volume_positions=lab_frame_points(volumes, toroidal_angle),
        optic_positions=lab_frame_points(optics_rz, toroidal_angle),
        volume_optic_index=jnp.where(is_hfs, 0, 1),
        collection_aperture_diameter=aperture_diameter,
        collection_focal_length=focal_length,
        fiber_core_diameter=fiber_core_diameter,
        name=name,
    )
    _check_scattering_angles(geometry, is_hfs)
    return geometry


def _check_scattering_angles(
    geometry: ts_types.ThomsonScatteringGeometry,
    is_hfs: jt.Bool[NpOrJaxArray, " n_volume"],
) -> None:
    """Reject a layout whose scattering angles are unusable; warn on a loose one.

    A traced layout is not checked, since it has no values to reject.
    """
    theta = ts_utils.volume_scattering_angles(geometry)
    if isinstance(theta, jax.core.Tracer):  # type: ignore
        return
    theta = np.degrees(np.asarray(theta))
    is_hfs = np.asarray(is_hfs)
    if theta.min() < THETA_MIN_DEG or theta.max() > THETA_MAX_DEG:
        raise ValueError(
            f"scattering angles {theta.min():.2f}-{theta.max():.2f} deg leave the "
            f"admissible band [{THETA_MIN_DEG}, {THETA_MAX_DEG}]; the imaged volume "
            f"length dL = w_fiber / (M sin theta) diverges on the beam axis. Move the "
            f"optics (`admissible_surface_angles` returns the valid surface positions)."
        )
    for label, served, limits in (
        ("low-angle", is_hfs, LOW_ANGLE_RANGE_DEG),
        ("high-angle", ~is_hfs, HIGH_ANGLE_RANGE_DEG),
    ):
        if not served.any():
            continue
        served_theta = theta[served]
        if served_theta.min() < limits[0] or served_theta.max() > limits[1]:
            print(
                f"[geometry_utils] WARNING: the {label} optic sees "
                f"{served_theta.min():.2f}-{served_theta.max():.2f} deg, outside its "
                f"range {limits}"
            )


def chords_from_surface_angles(
    surface: geometry_types.PoloidalSurface,
    surface_angles: jt.Float[NpOrJaxArray, "n_chord 2"],
    name_prefix: str = "chord",
) -> list[if_types.InterferometerGeometry]:
    """Chords as the segment between two points of `surface`, traceable in the angles.

    The general line of sight in one poloidal plane: two angles, no shared direction, so
    the chords of a layout need not be parallel and the endpoints stay a closed-form
    function of the design, which is what lets `jax.grad` reach a chord's two ends.

    Args:
        surface: The surface the endpoints sit on, which also fixes the plane.
        surface_angles: The two surface angles of each chord, in rad.
        name_prefix: Chord names are `<name_prefix>_<index>`.

    Returns:
        One geometry per chord, endpoints in the lab frame.
    """
    angles = jnp.reshape(jnp.asarray(surface_angles, dtype=float), (-1, 2))
    points = surface_point_at_angle(surface, jnp.reshape(angles, (-1,)))
    ends = lab_frame_points(points, surface.toroidal_angle).reshape(-1, 2, 3)
    return [
        if_utils.build_geometry_from_endpoints(
            ends[index, 0], ends[index, 1], name=f"{name_prefix}_{index}"
        )
        for index in range(ends.shape[0])
    ]


def chord_impact_parameters(
    chords: list[if_types.InterferometerGeometry],
    surface: geometry_types.PoloidalSurface,
    laser_surface_angles: tuple[float, float],
) -> jt.Float[np.ndarray, " n_chord"]:
    """Signed perpendicular offset of each chord from the centre, across the laser.

    Recovered from the endpoints rather than stored, so it stays right for chords that
    are not parallel; it is what orders a fan and what a layout table reports.

    Args:
        chords: Interferometer lines of sight of the layout.
        surface: The surface the layout is placed on.
        laser_surface_angles: The laser's two surface angles, which fix the normal.

    Returns:
        One signed impact parameter per chord, in m.
    """
    centre = np.asarray(surface.centre)
    ends = np.asarray(surface_point_at_angle(surface, np.asarray(laser_surface_angles)))
    tangent = ends[:, 1] - ends[:, 0]
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    impacts = []
    for chord in chords:
        start, end = _poloidal_endpoints(chord, surface.toroidal_angle)
        along = (end - start) / np.linalg.norm(end - start)
        foot = start + along * (along @ (centre - start))
        impacts.append(float(normal @ (foot - centre)))
    return np.asarray(impacts)


def _poloidal_endpoints(
    chord: if_types.InterferometerGeometry, toroidal_angle: float
) -> tuple[jt.Float[np.ndarray, " 2"], jt.Float[np.ndarray, " 2"]]:
    """A chord's lab-frame endpoints projected back into its poloidal plane."""
    cos_angle, sin_angle = np.cos(toroidal_angle), np.sin(toroidal_angle)
    start, end = np.asarray(chord.chord_start), np.asarray(chord.chord_end)
    return (
        np.array([float(start[0] * cos_angle + start[1] * sin_angle), float(start[2])]),
        np.array([float(end[0] * cos_angle + end[1] * sin_angle), float(end[2])]),
    )


def trim_chords_to_plasma(
    chords: list[if_types.InterferometerGeometry],
    coefficients: desc_types.DescFourierZernikeCoefficients,
    inward_shrink: float = 1e-6,
) -> list[if_types.InterferometerGeometry]:
    """Trim every chord to its in-plasma segment, so no line-integral sample is spent in
    vacuum, keeping the list length. Each chord lies in one poloidal plane with both
    endpoints outside the plasma, so the crossings are a 2D segment intersection
    (`segment_curve_span`, outermost pair). Fixed-boundary only, so the segment is fixed
    data computed once per run.

    Args:
        chords: The full surface-to-surface beams, as `chords_from_surface_angles`
            builds them.
        coefficients: Geometry whose `rho = 1` surface bounds the plasma.
        inward_shrink: Relative shrink keeping the endpoints inside the boundary.

    Returns:
        One geometry per input chord, in the same order.
    """
    trimmed = []
    for chord in chords:
        start = jnp.asarray(chord.chord_start)
        end = jnp.asarray(chord.chord_end)
        plasma = boundary_cross_section(coefficients, jnp.arctan2(start[1], start[0]))
        low, high, crosses = _segment_curve_crossings(
            plasma,
            jnp.stack([jnp.hypot(start[0], start[1]), start[2]]),
            jnp.stack([jnp.hypot(end[0], end[1]), end[2]]),
        )
        if not isinstance(crosses, jax.core.Tracer) and not bool(crosses):  # type: ignore
            print(
                f"[geometry_utils] WARNING: chord {chord.name} does not intersect the "
                f"plasma; keeping it untrimmed"
            )
            trimmed.append(chord)
            continue
        # A traced chord that misses the plasma keeps its whole length instead.
        low = jnp.where(crosses, low, 0.0)
        high = jnp.where(crosses, high, 1.0)
        middle = 0.5 * (low + high)
        half_width = 0.5 * (high - low) * (1.0 - inward_shrink)
        direction = end - start
        trimmed.append(
            if_utils.build_geometry_from_endpoints(
                start + (middle - half_width) * direction,
                start + (middle + half_width) * direction,
                name=f"{chord.name}_trimmed",
            )
        )
    return trimmed


def segment_curve_span(
    curve: jt.Float[NpOrJaxArray, "2 n_point"],
    start: jt.Float[NpOrJaxArray, " 2"],
    end: jt.Float[NpOrJaxArray, " 2"],
) -> tuple[float, float] | None:
    """Fractions along the segment `start -> end` where it crosses `curve`, or `None`.

    Returns the outermost pair of crossings, so a chord that leaves and re-enters a non-
    convex cross-section still has its whole plasma path bracketed; any interior vacuum
    gap stays correctly empty because the density there is zero anyway.
    """
    low, high, crosses = _segment_curve_crossings(curve, start, end)
    if not bool(crosses):
        return None
    return float(low), float(high)


def _segment_curve_crossings(
    curve: jt.Float[NpOrJaxArray, "2 n_point"],
    start: jt.Float[NpOrJaxArray, " 2"],
    end: jt.Float[NpOrJaxArray, " 2"],
) -> tuple[jt.Float[jax.Array, ""], jt.Float[jax.Array, ""], jt.Bool[jax.Array, ""]]:
    """`segment_curve_span` in fixed shapes, traceable in `start` and `end`: the
    outermost crossing fractions, and whether the segment crosses `curve` at all."""
    curve = jnp.asarray(curve)
    start = jnp.asarray(start)
    direction = jnp.asarray(end) - start
    length = jnp.linalg.norm(direction)
    safe_length = jnp.maximum(length, jnp.finfo(float).tiny)
    tangent = direction / safe_length
    normal = jnp.stack([-tangent[1], tangent[0]])
    side = normal @ (curve - start[:, None])
    side_next = jnp.roll(side, -1)
    crossing = ((side <= 0.0) & (side_next > 0.0)) | ((side_next <= 0.0) & (side > 0.0))
    # Every edge is evaluated and the non-crossing ones masked, so the shapes stay
    # fixed; the safe denominator keeps their gradients finite.
    fraction = side / jnp.where(crossing, side - side_next, 1.0)
    points = curve + fraction * (jnp.roll(curve, -1, axis=1) - curve)
    along = tangent @ (points - start[:, None]) / safe_length
    return (
        jnp.min(jnp.where(crossing, along, jnp.inf)),
        jnp.max(jnp.where(crossing, along, -jnp.inf)),
        jnp.any(crossing) & (length > 0.0),
    )
