import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import scipy.constants as sc
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (
    interferometer_types as if_types,
)

PHASE_UNCERTAINTY: float = 0.1
"""Constant 1-sigma phase uncertainty `sigma_phi` in rad, Dreier's conservative estimate
of the phase measurement error. Source: Dreier 2007, IPP Report 13/8, Sec. 3.2.3."""

PROBING_WAVELENGTH: float = 10.6e-6
"""CO2 probing wavelength `lambda` in m that converts `PHASE_UNCERTAINTY` to a line
density. Source: Dreier 2007, IPP Report 13/8, Sec. 3.2.1, Table 3.1."""

LINE_DENSITY_NOISE_M2: float = (
    4.0
    * math.pi
    * sc.epsilon_0
    * sc.electron_mass
    * sc.speed_of_light**2
    / (PROBING_WAVELENGTH * sc.elementary_charge**2)
    * PHASE_UNCERTAINTY
)
"""Default `sigma_IF` in m^-2, about 3.35e18: `PHASE_UNCERTAINTY` through the phase to
line-density relation of Dreier 2007, eq. (3.5). Absolute, the same on every chord."""


def interferometer_forward_signal(
    geometry: if_types.InterferometerGeometry,
    electron_density_fn: Callable[
        [jt.Float[NpOrJaxArray, "n_sample 3"]], jt.Float[NpOrJaxArray, " n_sample"]
    ],
    n_sample: int,
) -> jt.Float[jax.Array, ""]:
    """Line integral of ``ne(x, y, z)`` over one line of sight.

    Samples the chord (`chord_sample_points`), evaluates the density field
    `electron_density_fn` at those points, and integrates (`line_integrated_density_signal`).
    Differentiable through `electron_density_fn`, so it composes with the profile ->
    equilibrium -> flux-solve chain that provides ``ne(x, y, z)``.
    """
    sample_points = chord_sample_points(geometry, n_sample)
    electron_densities = electron_density_fn(sample_points)
    return line_integrated_density_signal(electron_densities, sample_points)


def compute_interferometer_uncertainty(
    line_integrated_density: NpOrJaxArray | ScalarFloat,
    sigma: ScalarFloat = LINE_DENSITY_NOISE_M2,
) -> jt.Float[jax.Array, "..."]:
    """Per-chord 1-sigma uncertainty in m^-2: the constant `sigma_IF` whatever the chord
    reads, the width `draw_recorded_line_integrated_densities` draws with.

    Args:
        line_integrated_density: Reading of each chord, which fixes only the shape.
        sigma: `sigma_IF` in m^-2, `LINE_DENSITY_NOISE_M2` by default.

    Returns:
        `sigma` at the shape of the readings.
    """
    return jnp.full(jnp.shape(line_integrated_density), sigma, dtype=float)


def draw_recorded_line_integrated_densities(
    generator: np.random.Generator,
    line_integrated_density: jt.Float[NpOrJaxArray, " n_chord"],
    sigma: ScalarFloat = LINE_DENSITY_NOISE_M2,
) -> jt.Float[np.ndarray, " n_chord"]:
    """Apply signal-independent Gaussian noise to the interferometer line-integrated
    electron density.

    The model follows Dreier (2007), Sec. 3.2.3. Dreier identifies the dominant
    interferometer uncertainty with the phase measurement, particularly the uncertainty
    in the zero crossings found when the noisy interference signal is converted into
    the rectangular signal the phase comparator reads. Since this error is independent
    of the magnitude of the phase shift, it is treated as constant background noise with
    a Gaussian distribution.

    Dreier adopts a conservative phase uncertainty of `sigma_phi = 0.1 rad`. Using the
    interferometer relation of Sec. 3.1.1, eq. (3.5),

        phi = lambda * e**2 / (4*pi*epsilon_0*m_e*c**2) * integral(n_e dl)

    the equivalent uncertainty in line-integrated electron density is

        sigma_IF = 4*pi*epsilon_0*m_e*c**2 / (lambda*e**2) * sigma_phi.

    For `lambda = 10.6 micrometers` and `sigma_phi = 0.1 rad`,

        sigma_IF ~= 3.35e18 m^-2,

    an absolute uncertainty, the same on every chord and independent of its reading.
    The synthetic measurement is therefore

        d_obs = d_true + epsilon,
        epsilon ~ Normal(0, sigma_IF**2),

    drawn independently per chord and never clipped. Host-side numpy, as the data path
    is.

    Args:
        generator: The seeded random generator.
        line_integrated_density: Noiseless `d_true` of each chord, in m^-2.
        sigma: `sigma_IF` in m^-2, `LINE_DENSITY_NOISE_M2` (from `PHASE_UNCERTAINTY`
            and `PROBING_WAVELENGTH`) by default.

    Returns:
        The recorded `d_obs` of each chord, in m^-2.

    References:
        H. Dreier, "Bayesian Experimental Design: Applications in Nuclear Fusion", IPP
        Report 13/8, Max-Planck-Institut fuer Plasmaphysik, 2007, Secs. 3.1.1 and 3.2.3.
    """
    expected = np.asarray(line_integrated_density, dtype=float)
    return expected + float(sigma) * generator.standard_normal(expected.shape)


def line_integrated_density_signal(
    electron_densities: jt.Float[NpOrJaxArray, " n_sample"],
    sample_points: jt.Float[NpOrJaxArray, "n_sample 3"],
) -> jt.Float[jax.Array, ""]:
    """Interferometer forward kernel: sampled densities to line-integrated density.

    Integrates the electron density along one chord with the trapezoidal rule over
    the chord arc length (cumulative 3D distance between consecutive
    `sample_points`), returning the single line-integrated density ``N``. Linear in
    `electron_densities`: a uniform density ``n`` over a chord of length ``L`` gives
    ``N = n L``. Pure and differentiable w.r.t. `electron_densities`.
    """
    arc_length = _chord_arc_length(sample_points)
    return jnp.trapezoid(jnp.asarray(electron_densities), arc_length)


def chord_sample_points(
    geometry: if_types.InterferometerGeometry,
    n_sample: int,
) -> jt.Float[jax.Array, "n_sample 3"]:
    """Points along a chord where the electron density is evaluated and integrated.

    `n_sample` points placed uniformly (endpoints included) on the straight line
    from `geometry.chord_start` to `geometry.chord_end`.
    """
    start = jnp.asarray(geometry.chord_start)
    end = jnp.asarray(geometry.chord_end)
    fractions = jnp.linspace(0.0, 1.0, n_sample)
    return start[None, :] + fractions[:, None] * (end - start)[None, :]


def build_geometry_from_endpoints(
    chord_start: jt.Float[NpOrJaxArray, "3"],
    chord_end: jt.Float[NpOrJaxArray, "3"],
    name: str,
) -> if_types.InterferometerGeometry:
    """Build a single-chord geometry from its lab-frame ``(x, y, z)`` endpoints."""
    return if_types.InterferometerGeometry(
        chord_start=jnp.asarray(chord_start),
        chord_end=jnp.asarray(chord_end),
        name=name,
    )


def _chord_arc_length(
    sample_points: jt.Float[NpOrJaxArray, "n_sample 3"],
) -> jt.Float[jax.Array, " n_sample"]:
    """Cumulative arc length along a chord from its sample positions.

    The arc-length coordinate of each sample is the running sum of the Euclidean
    distances between consecutive points, starting at zero: the integration variable
    for `jnp.trapezoid`.
    """
    positions = jnp.asarray(sample_points)
    segment_lengths = jnp.linalg.norm(jnp.diff(positions, axis=0), axis=-1)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(segment_lengths)])
