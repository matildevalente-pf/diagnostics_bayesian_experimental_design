# Thomson scattering forward model: spectrum, per-channel photoelectron counts, their
# uncertainties and the noise draw, in one place so data and likelihood cannot differ.

import functools
import math
from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import scipy.constants as sc
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    ScalarFloat,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_types as ts_types,
)

ELECTRON_MASS: float = sc.electron_mass
"""Electron rest mass in kg."""

ELEMENTARY_CHARGE: float = sc.elementary_charge
"""Elementary charge in C; also the J-per-eV conversion factor."""

SPEED_OF_LIGHT: float = sc.speed_of_light
"""Speed of light in m/s."""

ELECTRON_REST_ENERGY_EV: float = ELECTRON_MASS * SPEED_OF_LIGHT**2 / ELEMENTARY_CHARGE
"""Electron rest energy ``m_e c^2`` in eV (about 511 keV)."""

CLASSICAL_ELECTRON_RADIUS: float = sc.value("classical electron radius")
"""Classical electron radius ``r_e`` in m; ``r_e^2`` is the differential Thomson cross
section for light polarized perpendicular to the scattering plane (Hutchinson 2002,
equations 7.2.4-7.2.5, https://doi.org/10.1017/CBO9780511613630).
"""

PLANCK_TIMES_C: float = sc.Planck * SPEED_OF_LIGHT
"""``h c`` in J m; a photon of wavelength ``lambda`` carries ``h c / lambda``."""

VACUUM_PERMITTIVITY: float = sc.epsilon_0
"""Vacuum permittivity `epsilon_0` in F/m."""

SALPETER_PARAMETER_LIMIT: float = 0.1
"""Salpeter parameter at which `check_incoherent_scattering` raises, since the
incoherent spectrum of `thomson_signals` holds only well below one."""

W7X_POLYCHROMATOR_BANDS: tuple[tuple[float, float], ...] = (
    (750e-9, 920e-9),
    (920e-9, 1000e-9),
    (1000e-9, 1035e-9),
    (1035e-9, 1051e-9),
    (1051e-9, 1061e-9),
)
"""The five W7-X polychromator passbands as ``(low, high)`` edges in m, covering 10 eV
to 10 keV.

Source: Bozhenkov 2017, JINST 12 P10004, section 2,
https://doi.org/10.1088/1748-0221/12/10/P10004.
"""

W7X_LASER_WAVELENGTH: float = 1064e-9
"""Nd:YAG probing wavelength in m.

Source: Pasch 2016, Rev. Sci. Instrum. 87 11E729,
section II, https://doi.org/10.1063/1.4962248.
"""

W7X_LASER_PULSE_ENERGY: float = 2.4
"""Laser pulse energy in J, the top of the W7-X range of 0.7 to 2.4 J.

Source: Bozhenkov
2017, section 2, https://doi.org/10.1088/1748-0221/12/10/P10004.
"""

W7X_COLLECTION_APERTURE_DIAMETER: float = 0.16
"""Clear aperture of the W7-X collection lens in m.

Source: Pasch 2016, section II
(160 mm diameter, f/1.3), https://doi.org/10.1063/1.4962248.
"""

W7X_COLLECTION_FOCAL_LENGTH: float = 0.172
"""Focal length of the W7-X collection lens in m.

Source: Pasch 2016, section II
(f = 172 mm), https://doi.org/10.1063/1.4962248.
"""

W7X_FIBER_BUNDLE_DIAMETER: float = 3.0e-3
"""Diameter of the W7-X fiber bundles in m, used as the width imaged onto the laser.

Source: Pasch 2016, section II, and Bozhenkov 2017, section 2: 3 mm is the diameter of
the bundles at the polychromator end (numerical aperture 0.37); at the imaging end they
are 1.1 x 3.2 mm^2 rectangles.
"""

W7X_COLLECTION_MAGNIFICATION: float = 0.1
"""Design magnification of the W7-X collection optics, fixing the standoff a lens is
placed at (`collection_object_distance`).

Source: Pasch 2016, section II,
demagnification 7.8:1 at the core to 11.5:1 at the edge,
https://doi.org/10.1063/1.4962248.
"""

ThomsonNoiseModel = Literal["poisson", "electronic"]
"""Which uncertainty `signal_uncertainties` attaches to a reading: the photon-counting
limit `sqrt(D_i)` or the W7-X detector noise curve."""

ELECTRONIC_NOISE_A: float = 1.68
"""`a` of the detector noise curve `sigma / I = a exp(-b I) + c`, fitted at W7-X to
absolute-calibration data taken without plasma.

Approximate value read off Fig. 6 of Raak 2024, Bachelor thesis, TU Berlin, which
reports no numerical fit coefficients.
"""

ELECTRONIC_NOISE_B: float = 4.31e10
"""`b` of that noise curve in (V s)^-1, likewise an approximate value inferred from Fig.

6 of Raak 2024.
"""

ELECTRONIC_NOISE_C: float = 0.1
"""`c` of that noise curve, the relative uncertainty no amount of collected light beats.

Fig. 6 of Raak 2024 reads about 0.087. The 0.1 is this work's choice, raised to cover
part of the plasma background light, which section 3.2 of Raak 2024 treats instead as a
shift of the curve toward larger signals. Discussed with Golo Fuchert, 2026-09-25.
"""

DETECTION_SNR: float = 2.0
"""`I / sigma` a channel needs to count as detectable.

Source: Raak 2024, section 3.1,
where a reading below twice its standard deviation is set to zero.
"""

MINIMUM_DETECTABLE_INTEGRATED_SIGNAL: float = (
    -math.log((1.0 / DETECTION_SNR - ELECTRONIC_NOISE_C) / ELECTRONIC_NOISE_A)
    / ELECTRONIC_NOISE_B
)
"""`I_min` in V s, where the noise curve reaches `sigma / I = 1 / DETECTION_SNR`."""

NOISE_FLOOR_INTEGRATED_SIGNAL: float = (
    -math.log((1.0 - ELECTRONIC_NOISE_C) / ELECTRONIC_NOISE_A) / ELECTRONIC_NOISE_B
)
"""`I_floor` in V s, where the noise curve reaches `sigma / I = 1`, so that the absolute
electronic uncertainty of a vanishing signal stays at `I_floor`.

Source: Raak 2024, section 3.2, which raises `sigma / I` to one for small signals, and
the constant floor of its appendix A.7 code.
"""

MINIMUM_DETECTABLE_DENSITY: float = 5.0e18
"""Electron density in m^-3 of the weakest state the reference system is asked to
resolve, one of the two states `electronic_signal_scale` is calibrated at.

Inside the range of Pasch 2016, section I, whose lower bound is 2e18.
"""

MINIMUM_DETECTABLE_TEMPERATURE_EV: float = 20.0
"""Electron temperature in eV of that same weakest state, the lower limit the reference
system is specified for (Pasch 2016, section I, 20 eV to 10 keV)."""

CALIBRATION_BOUND: float = 1.05
"""`N` of the log-uniform prior `ln delta_j ~ U(-ln N, ln N)` on a volume's calibration
correction.

1.05 keeps the 5 percent Raman systematic of Bozhenkov 2017, section 4, as
the box edge; the box form was discussed with Golo Fuchert, 2026-09-25.
"""

MIN_EXPECTED_COUNTS: float = 10.0
"""Photoelectrons a channel must expect to stay in the likelihood under `poisson`, where
`sqrt(D_i)` is otherwise no Gaussian width.

The electronic models need no such guard,
their floor holding `sigma_i` finite. This work, no source.
"""

BAND_QUADRATURE_NODES: int = 16
"""Gauss-Legendre nodes per passband in `thomson_signals`.

Eight already agree with a 262 000-point reference to 7e-5; sixteen leave margin for the
narrowest cold spectra.
"""

MIN_BAND_EDGE: float = 700e-9
"""Bluest admissible passband edge in m.

Source: Bozhenkov 2017, section 2, the W7-X
collection optics are designed for 700 to 1064 nm,
https://doi.org/10.1088/1748-0221/12/10/P10004. Also Pasch section II. Design.
"""

MIN_BAND_WIDTH: float = 5e-9
"""Narrowest admissible passband in m.

No source, but discussed with Golo.
"""

FISHER_RULE_N_SCALE: int = 67
"""Candidate scales `fisher_band_scale` compares across `band_scale_bounds`, one per
0.01 of scale; the variance is flat around its minimum, so the grid decides nothing."""

_GAUSS_LEGENDRE_NODES, _GAUSS_LEGENDRE_WEIGHTS = (
    jnp.asarray(values)
    for values in np.polynomial.legendre.leggauss(BAND_QUADRATURE_NODES)
)


# Main caller that returns the D_i for each channel on each polychromator and
# corresponding uncertainties
def draw_recorded_signals(
    generator: np.random.Generator,
    signals: jt.Float[NpOrJaxArray, "n_volume n_channel"],
    uncertainties: jt.Float[NpOrJaxArray, "n_volume n_channel"],
    noise_model: ThomsonNoiseModel,
) -> jt.Float[np.ndarray, "n_volume n_channel"]:
    """One recording, `Poisson(D_i)` under `poisson` and `Normal(D_i, sigma_i^2)` under
    `electronic`, which draws about the noiseless signal and is never clipped at a
    detection threshold. Host-side numpy, as the data path is.

    Args:
        generator: The seeded random generator.
        signals: Expected signals `D_i`, in photoelectrons.
        uncertainties: `signal_uncertainties` of those signals, the Gaussian width.
        noise_model: Which uncertainty model the recording is drawn under.

    Returns:
        The recorded signals, in photoelectrons.
    """
    expected = np.maximum(np.asarray(signals), 0.0)
    if noise_model == "poisson":
        return generator.poisson(expected).astype(float)
    return expected + np.asarray(uncertainties) * generator.standard_normal(
        expected.shape
    )


# 1.1 Modeling of the Thomson scattering system - forward signals D_i


def thomson_signals(
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    electron_density: jt.Float[NpOrJaxArray, " n_volume"],
    electron_temperature_ev: jt.Float[NpOrJaxArray, " n_volume"],
    calibration: NpOrJaxArray | ScalarFloat,
) -> jt.Float[jax.Array, "n_volume n_channel"]:
    """Expected photoelectrons per channel and volume, eq. (1.2) of Bozhenkov 2017.

    D_i = calibration N_L n_e r_e^2 int_band (1 + eps) S deps, with
    N_L = E lambda_0 / (h c) the laser photons, unit transmission inside each passband
    and Gauss-Legendre quadrature per band. Sources:
    1. Bozhenkov 2017, JINST 12 P10004, eq. (1.2),
    https://doi.org/10.1088/1748-0221/12/10/P10004, the same formula as Merlo 2023,
    JINST 18 P11012, eq. (2.7), https://doi.org/10.1088/1748-0221/18/11/P11012.
    2. The top-hat transmission, one inside each passband and zero outside, idealizes
    the W7-X interference filters, which transmit T > 0.95 in band and block the laser
    line at OD 6: Pasch 2016, table I, https://doi.org/10.1063/1.4962248.

    Args:
        geometry: The layout, fixing every volume's scattering angle.
        response: Laser and passbands, scaled per volume by its `band_scales`.
        electron_density: Local electron density at every volume in m^-3.
        electron_temperature_ev: Local electron temperature at every volume in eV.
        calibration: `absolute_calibration`, one value per volume or one for all.

    Returns:
        `D[volume, channel]` in photoelectrons.
    """
    # Computation of the integral
    laser = response.laser_wavelength
    low, high = band_edges(response, response.band_scales)
    n_volume = jnp.asarray(electron_density).shape[0]
    low = jnp.broadcast_to(low / laser - 1.0, (n_volume, low.shape[-1]))
    high = jnp.broadcast_to(high / laser - 1.0, (n_volume, high.shape[-1]))
    half_width = 0.5 * (high - low)[..., None]
    nodes = 0.5 * (high + low)[..., None] + half_width * _GAUSS_LEGENDRE_NODES
    weights = half_width * _GAUSS_LEGENDRE_WEIGHTS
    spectra = jax.vmap(
        lambda eps, theta, te: spectral_density_relativistic(
            laser * (1.0 + eps), theta, te, laser
        )
    )(nodes, volume_scattering_angles(geometry), jnp.asarray(electron_temperature_ev))
    integral = jnp.sum(weights * (1.0 + nodes) * spectra, axis=-1)

    # Computa the prefactor of the equation
    laser_photons = response.laser_pulse_energy * laser / PLANCK_TIMES_C
    prefactor = (
        jnp.asarray(calibration)
        * laser_photons
        * CLASSICAL_ELECTRON_RADIUS**2
        * jnp.asarray(electron_density)
    )

    return prefactor[:, None] * integral


def spectral_density_relativistic(
    wavelength: NpOrJaxArray,
    scattering_angle: ScalarFloat,
    electron_temperature_ev: ScalarFloat,
    laser_wavelength: ScalarFloat,
) -> jt.Float[jax.Array, "..."]:
    """Relativistic incoherent spectral density S(eps, theta, 2 alpha) per unit eps.

    Zhuravlev's closed form times the (2,2) rational depolarization factor, for light
    polarized perpendicular to the scattering plane.
    Source: Naito 1993, Phys. Fluids B 5 4256, eqs. (3)-(5) and the (2,2) approximation
    on p. 4257, https://doi.org/10.1063/1.860593, with the sign of the zeta^6 term of p2
    from the erratum, Phys. Plasmas 1 806 (1994), https://doi.org/10.1063/1.870779.

    Args:
        wavelength: Scattered wavelengths in m, any shape.
        scattering_angle: Angle between the beam and the line of sight in rad.
        electron_temperature_ev: Electron temperature in eV, positive.
        laser_wavelength: Laser wavelength in m.

    Returns:
        `S` at every wavelength.
    """
    epsilon = jnp.asarray(wavelength) / laser_wavelength - 1.0
    two_alpha = ELECTRON_REST_ENERGY_EV / electron_temperature_ev
    return _zhuravlev_spectral_density(
        epsilon, scattering_angle, two_alpha
    ) * _depolarization_factor(epsilon, scattering_angle, two_alpha)


def volume_scattering_angles(
    geometry: ts_types.ThomsonScatteringGeometry,
) -> jt.Float[jax.Array, " n_volume"]:
    """Angle between the beam and each volume's line of sight to its optic, in rad.

    Args:
        geometry: The layout.

    Returns:
        One angle per volume; 0 is forward scattering, `pi` back scattering.
    """
    direction = jnp.asarray(geometry.laser_direction)
    direction = direction / jnp.linalg.norm(direction)
    volumes = jnp.asarray(geometry.volume_positions)
    optics = jnp.asarray(geometry.optic_positions)[
        jnp.asarray(geometry.volume_optic_index)
    ]
    scattered = optics - volumes
    scattered = scattered / jnp.linalg.norm(scattered, axis=-1, keepdims=True)
    return jnp.arccos(jnp.clip(scattered @ direction, -1.0, 1.0))


def salpeter_parameter(
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    electron_density: jt.Float[NpOrJaxArray, " n_volume"],
    electron_temperature_ev: jt.Float[NpOrJaxArray, " n_volume"],
) -> jt.Float[jax.Array, " n_volume"]:
    """Salpeter parameter `1 / (k lambda_D)` of every volume, with the scattering
    wavenumber `k = 4 pi sin(theta / 2) / lambda_0` of its scattering angle.

    Args:
        geometry: The layout, fixing every volume's scattering angle.
        response: Polychromator response, supplying the laser wavelength.
        electron_density: Local electron density at every volume in m^-3.
        electron_temperature_ev: Local electron temperature at every volume in eV.

    Returns:
        One value per volume.
    """
    debye_length = jnp.sqrt(
        VACUUM_PERMITTIVITY
        * jnp.asarray(electron_temperature_ev)
        / (ELEMENTARY_CHARGE * jnp.asarray(electron_density))
    )
    scattering_wavenumber = (
        4.0
        * jnp.pi
        * jnp.sin(0.5 * volume_scattering_angles(geometry))
        / response.laser_wavelength
    )
    return 1.0 / (scattering_wavenumber * debye_length)


def check_incoherent_scattering(
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    electron_density: jt.Float[NpOrJaxArray, " n_volume"],
    electron_temperature_ev: jt.Float[NpOrJaxArray, " n_volume"],
) -> None:
    """Raise if any volume reaches `SALPETER_PARAMETER_LIMIT`, where the incoherent
    spectrum of `thomson_signals` no longer applies. Host side, on concrete values.

    Args:
        geometry: The layout, fixing every volume's scattering angle.
        response: Polychromator response, supplying the laser wavelength.
        electron_density: Local electron density at every volume in m^-3.
        electron_temperature_ev: Local electron temperature at every volume in eV.

    Raises:
        ValueError: If a volume's Salpeter parameter is at or above the limit.
    """
    salpeter = np.asarray(
        salpeter_parameter(
            geometry, response, electron_density, electron_temperature_ev
        )
    )
    collective = np.flatnonzero(salpeter >= SALPETER_PARAMETER_LIMIT)
    if collective.size > 0:
        raise ValueError(
            f"Salpeter parameter up to {salpeter[collective].max():.3g} at volumes "
            f"{collective.tolist()}, at or above {SALPETER_PARAMETER_LIMIT}: the "
            "scattering is collective and the incoherent spectrum does not apply"
        )


def _naito_variables(
    epsilon: jt.Float[jax.Array, "..."], scattering_angle: ScalarFloat
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`x`, `u` and `y` of Naito 1993, defined with its equation (5)."""
    one_minus_cos = 1.0 - jnp.cos(scattering_angle)
    x = jnp.sqrt(1.0 + epsilon**2 / (2.0 * one_minus_cos * (1.0 + epsilon)))
    u = jnp.sin(scattering_angle) / one_minus_cos
    return x, u, 1.0 / jnp.sqrt(x**2 + u**2)


def _zhuravlev_spectral_density(
    epsilon: jt.Float[jax.Array, "..."],
    scattering_angle: ScalarFloat,
    two_alpha: ScalarFloat,
) -> jt.Float[jax.Array, "..."]:
    """S_Z of Naito 1993 equation (4), `K_2` by its large-argument expansion."""

    x, _, _ = _naito_variables(epsilon, scattering_angle)
    z = two_alpha
    bessel = jnp.sqrt(jnp.pi / (2.0 * z)) * (
        1.0 + 15.0 / (8.0 * z) + 105.0 / (128.0 * z**2) - 315.0 / (1024.0 * z**3)
    )
    one_minus_cos = 1.0 - jnp.cos(scattering_angle)

    return jnp.exp(-z * (x - 1.0)) / (
        2.0
        * bessel
        * (1.0 + epsilon) ** 3
        * jnp.sqrt(2.0 * one_minus_cos * (1.0 + epsilon) + epsilon**2)
    )


def _depolarization_factor(
    epsilon: jt.Float[jax.Array, "..."],
    scattering_angle: ScalarFloat,
    two_alpha: ScalarFloat,
) -> jt.Float[jax.Array, "..."]:
    """`q` of Naito 1993 equation (6), the (2,2) form of p. 4257 with the erratum."""
    x, _, y = _naito_variables(epsilon, scattering_angle)
    zeta = x * y
    eta = y / two_alpha
    p0 = 4.0 + 30.0 * zeta**2 - 55.0 * zeta**4
    p1 = -zeta * (24.0 - 545.0 * zeta**2 + 720.0 * zeta**4)
    p2 = 2.0 * (33.0 - 165.0 * zeta**2 + 240.0 * zeta**4 - 100.0 * zeta**6)
    q1 = 25.0 * zeta**3 * (29.0 - 42.0 * zeta**2)
    q2 = 5.0 * (18.0 - 66.0 * zeta**2 + 630.0 * zeta**4 - 805.0 * zeta**6)
    return 1.0 - 4.0 * eta * zeta * (p0 + p1 * eta + p2 * eta**2) / (
        p0 + q1 * eta + q2 * eta**2
    )


# 1.2 Modeling of the Thomson scattering system - forward signals D_i - abs calibration


def collection_object_distance(
    focal_length: ScalarFloat, magnification: ScalarFloat
) -> float:
    """Standoff at which a thin lens images with `magnification`: `f (1 + 1 / M)`.

    Source: Pasch 2016, section II, https://doi.org/10.1063/1.4962248, whose 172 mm
    optics demagnify 7.8:1 to 11.5:1, i.e. they stand 1.5 to 2.2 m from the volumes.

    Args:
        focal_length: Focal length of the collection lens in m.
        magnification: Image size over object size.

    Returns:
        The lens-to-volume distance in m.
    """
    return float(focal_length) * (1.0 + 1.0 / float(magnification))


def collection_optics(
    geometry: ts_types.ThomsonScatteringGeometry,
) -> tuple[jt.Float[jax.Array, " n_volume"], jt.Float[jax.Array, " n_volume"]]:
    """Imaged volume length dL and collection solid angle dOmega of every volume.

    dL = w_fiber / (M sin(theta)). Given by the:
     - geometry (theta) and standoff (d) -> unique to each volume element.
     - lens magnification M = f / (d - f), where f is the focal length.
     - width of the fiber in the optical system.

    dOmega = pi (D / 2)^2 / d^2, where D is the aperture diameter of the lens.

    Both expressions are elementary Gaussian optics, taken from no paper:
    `M = f / (d - f)` solves the thin-lens equation for an object at `d`, and `dOmega`
    is the small-angle solid angle of a circular aperture of area `pi (D / 2)^2`.

    Sources:
    - The values D = 160 mm, f = 172 mm ("[160 mm diameter, f = 172 mm, f/1.3,
      numerical aperture (NA) = 0.37]") and `dL` as the fiber diameter over the
      demagnification, which is how the reference system's own volume lengths are
      quoted: Pasch 2016, section II, https://doi.org/10.1063/1.4962248. Its 25 mm core
      and 37 mm edge volumes come from the 3.2 mm long side of its 1.1 x 3.2 mm^2
      bundles at 7.8:1 and 11.5:1; the 3 mm used here gives 23.4 and 34.5 mm.
    - That these two factors are what turns a scattered spectrum into a collected
      signal, `N_s = N_i r_e^2 dOmega n_e dL` (eq. 7.2.34), and that the `sin theta`
      belongs in `dL` because the beam length seen perpendicular to the viewing
      direction is `dL sin theta` (p. 288, and p. 289 names `dOmega dL sin theta` the
      etendue): Hutchinson 2002, section 7.2.4,
      https://doi.org/10.1017/CBO9780511613630.

    Args:
        geometry: The layout, including the lens parameters.

    Returns:
        `dL` in m and `dOmega` in sr, per volume.

    Raises:
        ValueError: If a volume lies within the focal length of its lens, where no image
            forms.
    """
    positions = jnp.asarray(geometry.volume_positions)
    optics = jnp.asarray(geometry.optic_positions)[
        jnp.asarray(geometry.volume_optic_index)
    ]
    standoff = jnp.linalg.norm(positions - optics, axis=-1)
    focal_length = geometry.collection_focal_length
    if not isinstance(standoff, jax.core.Tracer) and bool(  # type: ignore
        jnp.any(standoff <= focal_length)
    ):
        raise ValueError(
            f"standoffs of {float(jnp.min(standoff)):.3f} m do not exceed the focal "
            f"length {float(focal_length):.3f} m, so the lens forms no image of the "
            f"volume; place the optic at `collection_object_distance`."
        )
    magnification = focal_length / (standoff - focal_length)
    volume_length = geometry.fiber_core_diameter / (
        magnification * jnp.sin(volume_scattering_angles(geometry))
    )
    solid_angle = (
        jnp.pi * (geometry.collection_aperture_diameter / 2.0) ** 2 / standoff**2
    )
    return volume_length, solid_angle


def absolute_calibration(
    geometry: ts_types.ThomsonScatteringGeometry,
) -> jt.Float[jax.Array, " n_volume"]:
    """`dL dOmega` per volume, the computed twin of a measured `dL dOmega g0`.

    Source: Bozhenkov 2017, section 4, https://doi.org/10.1088/1748-0221/12/10/P10004,
    calibrates this factor per volume by Raman scattering. A designed instrument
    computes it from its optics (`collection_optics`) instead.

    Args:
        geometry: The layout, including the lens parameters.

    Returns:
        The calibration in m sr per collected photon, per volume.
    """
    volume_length, solid_angle = collection_optics(geometry)
    return jnp.asarray(volume_length * solid_angle)


def effective_calibration(
    nominal: jt.Float[NpOrJaxArray, " n_volume"],
    corrections: jt.Float[NpOrJaxArray, " n_volume"] | None,
) -> jt.Float[jax.Array, " n_volume"]:
    """`C_j^eff = C_j^geom delta_j`, the geometric calibration times its correction.

    One factor per volume, so it scales every spectral channel of that polychromator
    alike and leaves the other volumes untouched.

    Args:
        nominal: `absolute_calibration`, the geometric value of every volume.
        corrections: One `delta_j` per volume; None keeps the geometric value.

    Returns:
        The calibration `thomson_signals` is evaluated with, per volume.
    """
    if corrections is None:
        return jnp.asarray(nominal)
    return jnp.asarray(nominal) * jnp.asarray(corrections)


# 2. Uncertainty model of the detector


def signal_uncertainties(
    signals: jt.Float[NpOrJaxArray, "n_volume n_channel"],
    noise_model: ThomsonNoiseModel,
    signal_scale: ScalarFloat,
    min_expected_counts: ScalarFloat = MIN_EXPECTED_COUNTS,
) -> jt.Float[jax.Array, "n_volume n_channel"]:
    """One standard deviation per channel under one of the two noise models.

    `poisson` is the photon-counting limit `sqrt(D_i)`, `electronic` the W7-X detector
    noise curve of Raak 2024, Fig. 6, without its plasma background amplification.

    Args:
        signals: Expected signals `D_i`, in photoelectrons.
        noise_model: Which of the two uncertainty models to evaluate.
        signal_scale: `kappa` in V s per photoelectron, from `electronic_signal_scale`.
        min_expected_counts: Channels expecting fewer photoelectrons get `sigma = 0`,
            under `poisson` only.

    Returns:
        One standard deviation per channel, in photoelectrons.
    """
    expected = jnp.maximum(jnp.asarray(signals), 0.0)
    if noise_model == "poisson":
        return jnp.where(expected >= min_expected_counts, jnp.sqrt(expected), 0.0)
    # `sigma / I = a exp(-b I) + c` at `I = kappa D`, times `D` to undo the
    # normalization, and held at the floor where the curve reaches `sigma / I = 1`.
    relative = (
        ELECTRONIC_NOISE_A * jnp.exp(-ELECTRONIC_NOISE_B * signal_scale * expected)
        + ELECTRONIC_NOISE_C
    )
    return jnp.maximum(
        expected * relative, NOISE_FLOOR_INTEGRATED_SIGNAL / signal_scale
    )


def electronic_signal_scale(
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    calibration: NpOrJaxArray | ScalarFloat,
    ne_min: ScalarFloat = MINIMUM_DETECTABLE_DENSITY,
    te_min_ev: ScalarFloat = MINIMUM_DETECTABLE_TEMPERATURE_EV,
) -> float:
    """`kappa = I_min / D_min^ref` in V s per photoelectron, tying the synthetic signal
    scale to the electrical scale of the Raak 2024 noise curve. Calibrate it once on the
    reference configuration, since a different layout must not move the detector.

    Args:
        geometry: The reference layout, fixing every volume's angle and standoff.
        response: Laser and passbands of the reference system.
        calibration: `absolute_calibration` of that reference layout.
        ne_min: Electron density of the minimum detectable reference state in m^-3.
        te_min_ev: Electron temperature of that state in eV.

    Returns:
        The conversion factor `kappa`.
    """
    n_volume = jnp.asarray(geometry.volume_positions).shape[0]
    reference_signals = thomson_signals(
        geometry,
        response,
        jnp.full(n_volume, ne_min),
        jnp.full(n_volume, te_min_ev),
        calibration,
    )
    # Raak 2024, section 3.1: two channels must read a signal for the spectrum to be
    # fitted, so a polychromator's second largest reading is what it has to detect.
    second_largest = jnp.sort(reference_signals, axis=1)[:, -2]
    return float(MINIMUM_DETECTABLE_INTEGRATED_SIGNAL / jnp.min(second_largest))


# 3. Selection of the polychromator passbands


def build_polychromator_response(
    polychromator_bands: list[tuple[float, float]],
    laser_wavelength: float,
    laser_pulse_energy: float,
) -> ts_types.PolychromatorResponse:
    """The response of a polychromator with the given `(low, high)` passbands in m.

    Args:
        polychromator_bands: Passband edges, `W7X_POLYCHROMATOR_BANDS` for W7-X.
        laser_wavelength: Laser wavelength in m.
        laser_pulse_energy: Laser pulse energy in J.

    Returns:
        The response, with the bands stored as centres and half-widths.
    """
    bands = np.asarray(polychromator_bands, dtype=np.float64)
    return ts_types.PolychromatorResponse(
        laser_wavelength=laser_wavelength,
        laser_pulse_energy=laser_pulse_energy,
        filter_centers=0.5 * (bands[:, 0] + bands[:, 1]),
        filter_widths=0.5 * (bands[:, 1] - bands[:, 0]),
    )


def band_edges(
    response: ts_types.PolychromatorResponse,
    band_scales: jt.Float[NpOrJaxArray, " n_volume"] | None = None,
) -> tuple[jt.Float[jax.Array, "..."], jt.Float[jax.Array, "..."]]:
    """`(low, high)` passband edges in m, scaled per volume when `band_scales` is given.

    A scale `s` moves every edge's distance to the laser line together,
    `edge' = lambda_0 - s (lambda_0 - edge)`, so the set follows the width of the
    spectrum without changing its relative structure. This work's parametrization.

    Args:
        response: The stored passbands and the laser wavelength.
        band_scales: One factor per volume; None returns the stored edges.

    Returns:
        Edges of shape `(n_channel,)`, or `(n_volume, n_channel)` when scaled.
    """
    centers = jnp.asarray(response.filter_centers)
    widths = jnp.asarray(response.filter_widths)
    low, high = centers - widths, centers + widths
    if band_scales is None:
        return jnp.asarray(low), jnp.asarray(high)
    scale = jnp.asarray(band_scales)[:, None]
    laser = response.laser_wavelength
    return (
        jnp.asarray(laser - scale * (laser - low)),
        jnp.asarray(laser - scale * (laser - high)),
    )


def band_scale_bounds(
    response: ts_types.PolychromatorResponse,
) -> tuple[float, float]:
    """Admissible `(low, high)` band-scale interval of `band_edges` for this response.

    The low bound keeps the narrowest channel `MIN_BAND_WIDTH` wide, the high bound
    keeps every edge above `MIN_BAND_EDGE` (Bozhenkov 2017, section 2,
    https://doi.org/10.1088/1748-0221/12/10/P10004).

    Args:
        response: The stored passbands and the laser wavelength.

    Returns:
        The lowest and highest admissible scale.
    """
    # Evaluated eagerly even inside a trace, since the stored passbands are fixed data.
    with jax.ensure_compile_time_eval():
        low, high = band_edges(response)
        narrowest = float(jnp.min(high - low))
        bluest = float(jnp.min(low))
    laser = float(response.laser_wavelength)
    return MIN_BAND_WIDTH / narrowest, (laser - MIN_BAND_EDGE) / (laser - bluest)


def fisher_band_scale(
    electron_density: jt.Float[NpOrJaxArray, " n_volume"],
    electron_temperature_ev: jt.Float[NpOrJaxArray, " n_volume"],
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    calibration: NpOrJaxArray | ScalarFloat,
    noise_model: ThomsonNoiseModel,
    signal_scale: ScalarFloat,
) -> jt.Float[jax.Array, " n_volume"]:
    """Band scale per volume minimizing the Cramer-Rao variance of `ln Te` at the
    expected state, the locally c-optimal design with `c = (0, 1)` (Lopez-Fidalgo 2023,
    section 2.4.12, https://doi.org/10.1007/978-3-031-35918-7), searched over
    `band_scale_bounds`.

    Args:
        electron_density: Expected electron density at every volume in m^-3.
        electron_temperature_ev: Expected electron temperature at every volume in eV.
        geometry: The layout, fixing every volume's scattering angle.
        response: The stored passbands and the laser wavelength.
        calibration: `absolute_calibration`, as for `thomson_signals`.
        noise_model: Which uncertainty model weights the Fisher information.
        signal_scale: `kappa`, as for `signal_uncertainties`.

    Returns:
        The scale factor per volume, one of `FISHER_RULE_N_SCALE` admissible values.
    """
    low, high = band_scale_bounds(response)
    scales = jnp.linspace(low, high, FISHER_RULE_N_SCALE)
    log_state = jnp.stack(
        [
            jnp.log(jnp.asarray(electron_density, dtype=float)),
            jnp.log(jnp.asarray(electron_temperature_ev, dtype=float)),
        ]
    )
    variance = jax.vmap(
        functools.partial(
            _log_temperature_variance,
            log_state=log_state,
            geometry=geometry,
            response=response,
            calibration=calibration,
            noise_model=noise_model,
            signal_scale=signal_scale,
        )
    )(scales)
    return scales[jnp.argmin(variance, axis=0)]


def log_temperature_variance(
    log_state: jt.Float[jax.Array, "2 n_volume"],
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    calibration: NpOrJaxArray | ScalarFloat,
    noise_model: ThomsonNoiseModel,
    signal_scale: ScalarFloat,
) -> jt.Float[jax.Array, " n_volume"]:
    """Cramer-Rao variance of `ln Te` per volume, `[F^-1]_22` of the Fisher information
    `F = J^T diag(1/sigma^2) J`, `J = dD / d(ln ne, ln Te)`, of one volume's channels
    (Lopez-Fidalgo 2023, Theorem 3.2 and Remark 3.3) under the likelihood
    `signal_uncertainties` defines.

    Args:
        log_state: `ln ne` over `ln Te`, one column per volume.
        geometry: The layout, fixing every volume's scattering angle.
        response: The passbands as stored, at their per-volume `band_scales`.
        calibration: `absolute_calibration`, as for `thomson_signals`.
        noise_model: Which uncertainty model weights the Fisher information.
        signal_scale: `kappa`, as for `signal_uncertainties`.

    Returns:
        The variance per volume, infinite where `ne` and `Te` cannot be separated.
    """
    forward = functools.partial(
        _signals_of_log_state,
        geometry=geometry,
        response=response,
        calibration=calibration,
    )
    n_volume = log_state.shape[1]
    ones, zeros = jnp.ones(n_volume), jnp.zeros(n_volume)
    signals, d_density = jax.jvp(forward, (log_state,), (jnp.stack([ones, zeros]),))
    _, d_temperature = jax.jvp(forward, (log_state,), (jnp.stack([zeros, ones]),))
    sigma = signal_uncertainties(signals, noise_model, signal_scale)
    weight = jnp.where(
        sigma > 0.0, 1.0 / jnp.square(jnp.where(sigma > 0.0, sigma, 1.0)), 0.0
    )
    f_nn = jnp.sum(weight * jnp.square(d_density), axis=1)
    f_nt = jnp.sum(weight * d_density * d_temperature, axis=1)
    f_tt = jnp.sum(weight * jnp.square(d_temperature), axis=1)
    determinant = f_nn * f_tt - jnp.square(f_nt)
    return jnp.where(determinant > 0.0, f_nn / determinant, jnp.inf)


def _log_temperature_variance(
    scale: ScalarFloat,
    *,
    log_state: jt.Float[jax.Array, "2 n_volume"],
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    calibration: NpOrJaxArray | ScalarFloat,
    noise_model: ThomsonNoiseModel,
    signal_scale: ScalarFloat,
) -> jt.Float[jax.Array, " n_volume"]:
    """`log_temperature_variance` with one band scale on every volume, the candidate
    `fisher_band_scale` compares."""
    n_volume = log_state.shape[1]
    scaled = response.model_copy(update={"band_scales": jnp.full(n_volume, scale)})
    return log_temperature_variance(
        log_state, geometry, scaled, calibration, noise_model, signal_scale
    )


def _signals_of_log_state(
    log_state: jt.Float[jax.Array, "2 n_volume"],
    *,
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    calibration: NpOrJaxArray | ScalarFloat,
) -> jt.Float[jax.Array, "n_volume n_channel"]:
    """`thomson_signals` of stacked `(ln ne, ln Te)` rows, the state Jacobians use."""
    return thomson_signals(
        geometry, response, jnp.exp(log_state[0]), jnp.exp(log_state[1]), calibration
    )


# 4. Local inversion: one volume's `ne` and `Te` from its own channels


LOCAL_FIT_MAX_TEMPERATURE_EV: float = 1.0e5
"""Hottest state the local inversion brackets, well above the 13 keV axis of the hottest
training equilibrium at the machine scale."""

LOCAL_FIT_N_TEMPERATURE: int = 201
"""Temperatures in that bracket, log-spaced from `MINIMUM_DETECTABLE_TEMPERATURE_EV`."""

LOCAL_FIT_NEWTON_STEPS: int = 20
"""Gauss-Newton steps taken from the bracketed start, as the global MAP takes."""


class LocalStateEstimate(NamedTuple):
    """One scattering volume's own `ne` and `Te`, with the 1-sigma width of each."""

    electron_density: jt.Float[jax.Array, " n_volume"]
    """Fitted electron density in m^-3; NaN where the channels fix no state."""

    electron_density_sigma: jt.Float[jax.Array, " n_volume"]
    """1-sigma width of that density in m^-3."""

    electron_temperature_ev: jt.Float[jax.Array, " n_volume"]
    """Fitted electron temperature in eV; NaN wherever the density is."""

    electron_temperature_sigma_ev: jt.Float[jax.Array, " n_volume"]
    """1-sigma width of that temperature in eV."""


def local_state_estimate(
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    signals: jt.Float[NpOrJaxArray, "n_volume n_channel"],
    uncertainties: jt.Float[NpOrJaxArray, "n_volume n_channel"],
    calibration: NpOrJaxArray | ScalarFloat,
) -> LocalStateEstimate:
    """Maximum-likelihood `(ne, Te)` of every volume from that volume's own channels,
    and the 1-sigma width of each: the per-volume evaluation of Bozhenkov 2017,
    section 5, https://doi.org/10.1088/1748-0221/12/10/P10004.

    Args:
        geometry: The layout the signals were recorded on, fixing the angles.
        response: Laser and passbands, including the per-volume band scales.
        signals: Recorded signals `D_i` per volume and channel, in photoelectrons.
        uncertainties: Their 1-sigma widths; zero or non-finite masks a channel.
        calibration: The calibration the signals are read with, as `thomson_signals`
            takes it.

    Returns:
        The density, the temperature and the width of each, one value per volume.
    """
    observed = jnp.asarray(signals)
    sigma = jnp.asarray(uncertainties)
    # The same channels `likelihood_utils` keeps, weighted the same way.
    valid = jnp.isfinite(observed) & jnp.isfinite(sigma) & (sigma > 0.0)
    weight = jnp.where(valid, 1.0 / jnp.square(jnp.where(valid, sigma, 1.0)), 0.0)
    observed = jnp.where(valid, observed, 0.0)

    n_volume = observed.shape[0]
    forward = functools.partial(
        _signals_of_log_state,
        geometry=geometry,
        response=response,
        calibration=calibration,
    )
    log_state = _bracketed_log_state(forward, observed, weight, n_volume)
    step = functools.partial(
        _local_newton_step, forward=forward, observed=observed, weight=weight
    )
    for _ in range(LOCAL_FIT_NEWTON_STEPS):
        log_state = step(log_state)

    # The Laplace covariance at the optimum, the inverse of the same Fisher information
    # the band-scale rule minimizes; a volume too poorly measured to invert has none.
    fisher, _ = _local_normal_equations(log_state, forward, observed, weight)
    regular = jnp.linalg.det(fisher) > 0.0
    covariance = jnp.linalg.inv(jnp.where(regular[:, None, None], fisher, jnp.eye(2)))
    # Fitted in logs, so the diagonal is a relative width; back to SI by the delta
    # method, `sigma_x = x sigma_(ln x)`.
    relative = jnp.sqrt(jnp.diagonal(covariance, axis1=1, axis2=2))
    density = jnp.exp(log_state[0])
    temperature = jnp.exp(log_state[1])
    return LocalStateEstimate(
        electron_density=jnp.where(regular, density, jnp.nan),
        electron_density_sigma=jnp.where(regular, density * relative[:, 0], jnp.nan),
        electron_temperature_ev=jnp.where(regular, temperature, jnp.nan),
        electron_temperature_sigma_ev=jnp.where(
            regular, temperature * relative[:, 1], jnp.nan
        ),
    )


def _bracketed_log_state(
    forward: Callable[
        [jt.Float[jax.Array, "2 n_volume"]], jt.Float[jax.Array, "n_volume n_channel"]
    ],
    observed: jt.Float[jax.Array, "n_volume n_channel"],
    weight: jt.Float[jax.Array, "n_volume n_channel"],
    n_volume: int,
) -> jt.Float[jax.Array, "2 n_volume"]:
    """The best `(ln ne, ln Te)` of every volume on a log grid of temperatures, which is
    where its Gauss-Newton starts."""
    temperatures = jnp.geomspace(
        MINIMUM_DETECTABLE_TEMPERATURE_EV,
        LOCAL_FIT_MAX_TEMPERATURE_EV,
        LOCAL_FIT_N_TEMPERATURE,
    )
    # `D` is exactly linear in `ne`, so a temperature fixes the least-squares density in
    # closed form and the grid has only the one remaining direction to search.
    unit_density_signals = jax.vmap(forward)(
        jnp.stack(
            [
                jnp.zeros((LOCAL_FIT_N_TEMPERATURE, n_volume)),
                jnp.broadcast_to(
                    jnp.log(temperatures)[:, None],
                    (LOCAL_FIT_N_TEMPERATURE, n_volume),
                ),
            ],
            axis=1,
        )
    )
    curvature = jnp.sum(weight * jnp.square(unit_density_signals), axis=-1)
    density = jnp.where(
        curvature > 0.0,
        jnp.sum(weight * unit_density_signals * observed, axis=-1)
        / jnp.where(curvature > 0.0, curvature, 1.0),
        MINIMUM_DETECTABLE_DENSITY,
    )
    misfit = jnp.sum(
        weight * jnp.square(density[..., None] * unit_density_signals - observed),
        axis=-1,
    )
    best = jnp.argmin(misfit, axis=0)
    return jnp.stack(
        [
            jnp.log(
                jnp.maximum(
                    density[best, jnp.arange(n_volume)], MINIMUM_DETECTABLE_DENSITY
                )
            ),
            jnp.log(temperatures[best]),
        ]
    )


def _local_newton_step(
    log_state: jt.Float[jax.Array, "2 n_volume"],
    *,
    forward: Callable[
        [jt.Float[jax.Array, "2 n_volume"]], jt.Float[jax.Array, "n_volume n_channel"]
    ],
    observed: jt.Float[jax.Array, "n_volume n_channel"],
    weight: jt.Float[jax.Array, "n_volume n_channel"],
) -> jt.Float[jax.Array, "2 n_volume"]:
    """One Gauss-Newton step of every volume's two-parameter fit, taken at once."""
    fisher, gradient = _local_normal_equations(log_state, forward, observed, weight)
    # A volume whose channels cannot fix both directions keeps the state it has.
    regular = jnp.linalg.det(fisher) > 0.0
    step = jnp.linalg.solve(
        jnp.where(regular[:, None, None], fisher, jnp.eye(2)),
        jnp.where(regular[:, None], gradient, 0.0)[..., None],
    )
    return log_state - step[..., 0].T


def _local_normal_equations(
    log_state: jt.Float[jax.Array, "2 n_volume"],
    forward: Callable[
        [jt.Float[jax.Array, "2 n_volume"]], jt.Float[jax.Array, "n_volume n_channel"]
    ],
    observed: jt.Float[jax.Array, "n_volume n_channel"],
    weight: jt.Float[jax.Array, "n_volume n_channel"],
) -> tuple[jt.Float[jax.Array, "n_volume 2 2"], jt.Float[jax.Array, "n_volume 2"]]:
    """`J^T W J` and `J^T W r` of every volume over its own channels, with `J = dD /
    d(ln ne, ln Te)`; the Fisher information of `_log_temperature_variance` and the
    weighted misfit gradient beside it."""
    n_volume = log_state.shape[1]
    ones, zeros = jnp.ones(n_volume), jnp.zeros(n_volume)
    predicted, d_density = jax.jvp(forward, (log_state,), (jnp.stack([ones, zeros]),))
    _, d_temperature = jax.jvp(forward, (log_state,), (jnp.stack([zeros, ones]),))
    jacobian = jnp.stack([d_density, d_temperature], axis=1)
    weighted = weight[:, None, :] * jacobian
    return (
        weighted @ jnp.swapaxes(jacobian, 1, 2),
        jnp.sum(weighted * (predicted - observed)[:, None, :], axis=-1),
    )
