import math
from collections.abc import Callable

import jax.numpy as jnp
import jaxtyping as jt
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes as mpl_axes
from matplotlib import figure as mpl_figure
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
    geometry_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    interferometer_types as if_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    interferometer_utils as if_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_types as ts_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_utils as ts,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E501
    pubfig,
)

OPTIC_SERIES: tuple[tuple[str, tuple[float, float], str], ...] = (
    ("low-angle, views HFS", geometry_utils.LOW_ANGLE_RANGE_DEG, "tab:orange"),
    ("high-angle, views LFS", geometry_utils.HIGH_ANGLE_RANGE_DEG, "tab:blue"),
)
"""Name, admissible scattering-angle range and colour of each optic, by its index.

The index is `ThomsonScatteringGeometry.volume_optic_index`, which
`geometry_utils.thomson_volumes_on_chord` assigns as `0` for the high-field-side
volumes and `1` for the low-field-side ones.
"""


def plot_chord_layout(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    geometry: ts_types.ThomsonScatteringGeometry,
    signals: jt.Float[NpOrJaxArray, "n_volume n_channel"],
    uncertainties: jt.Float[NpOrJaxArray, "n_volume n_channel"],
    surface_offset: float = geometry_utils.DEFAULT_SURFACE_OFFSET,
    show_sight_lines: bool = True,
    show_admissible: bool = True,
    axes_pair: tuple = (),
    active_volume_indices: tuple[int, ...] | None = None,
) -> tuple[mpl_figure.Figure, mpl_figure.Figure]:
    """Two-panel Thomson chord layout: cross-section geometry and collected light.

    Left: the plasma boundary, the auxiliary design surface, the laser path, the port of
    each optic where its line of sight crosses the design surface (the lens sits beyond
    it at its design standoff, outside the frame), and the scattering volumes coloured
    by the light they collect.
    Right: the collected light along the chord with its error bars, the chord/LCFS
    crossings and the magnetic axis. A lens whose served volumes fall outside its
    scattering-angle limits is ringed in red, which is how a violated layout announces
    itself.

    Draws what it is given and runs no forward model: the light is the channel sum of
    `signals` and the bars are `uncertainties` added in quadrature, so a caller that
    passes the target's own arrays gets the bars the likelihood used.

    Args:
        coefficients: DESC geometry coefficients of the equilibrium.
        geometry: The layout the signals were computed on, which fixes the volumes,
            the lenses, the laser and which lens serves which volume.
        signals: Per-channel signals, `synthetic_target_utils.readings_on_layout`.
        uncertainties: Their uncertainties, zero for a masked channel.
        surface_offset: Distance of the design surface outside the plasma in m.
        show_sight_lines: Draw the volume-to-optic sight lines.
        show_admissible: Draw the admissible surface angles of each optic set.
        axes_pair: Optional pair of axes to draw onto instead of a fresh figure.
        active_volume_indices: Volumes a likelihood actually reads; every other one is
            crossed out as masked. None marks none, which is the layout as built.

    Returns:
        The two panels' figures (the same object twice when they share one figure).

    Raises:
        ValueError: For a geometry with more optics than `OPTIC_SERIES` describes.
    """
    toroidal_angle = float(geometry.reference_phi)
    optics = _poloidal(geometry.optic_positions, toroidal_angle)
    if optics.shape[1] > len(OPTIC_SERIES):
        raise ValueError(
            f"{optics.shape[1]} optics but OPTIC_SERIES describes "
            f"{len(OPTIC_SERIES)}; this figure is built for the two-optic layout."
        )
    plasma, surface = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, surface_offset
    )
    surface_curve, centre = np.asarray(surface.curve), np.asarray(surface.centre)
    axis = geometry_utils.magnetic_axis(coefficients, toroidal_angle)
    volumes = _poloidal(geometry.volume_positions, toroidal_angle)
    direction = _poloidal(
        np.asarray(geometry.laser_direction)[None, :], toroidal_angle
    )[:, 0]
    direction = direction / np.linalg.norm(direction)
    # The geometry stores where the beam enters and which way it goes, so the far end is
    # the second crossing of the design surface along that ray.
    beam_start = _poloidal(np.asarray(geometry.laser_origin)[None, :], toroidal_angle)[
        :, 0
    ]
    reach = 2.0 * float(np.max(np.linalg.norm(surface_curve - centre[:, None], axis=0)))
    surface_span = geometry_utils.segment_curve_span(
        surface_curve, beam_start, beam_start + reach * direction
    )
    beam_end = beam_start + (
        1.0 if surface_span is None else surface_span[1]
    ) * reach * (direction)
    span = geometry_utils.segment_curve_span(plasma, beam_start, beam_end)
    along = direction @ (volumes - centre[:, None])
    n_elements = volumes.shape[1]

    signals = np.asarray(signals)
    light = signals.sum(axis=1)
    sigma = np.sqrt((np.asarray(uncertainties) ** 2).sum(axis=1))
    optic_index = np.asarray(geometry.volume_optic_index)

    if axes_pair:
        axes, axes_light = axes_pair
    else:
        _, (axes, axes_light) = plt.subplots(
            1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.0, 1.15]}
        )
    axes.plot(*plasma, "-", color="k", lw=1.6, label="plasma boundary (rho = 1)")
    axes.plot(
        *surface_curve,
        "--",
        color="tab:red",
        lw=1.4,
        label=f"auxiliary design surface (+{surface_offset:.3f} m)",
    )
    axes.plot(
        *np.stack([beam_start, beam_end], axis=1),
        "-",
        color="0.55",
        lw=1.4,
        zorder=2,
        label="laser path",
    )

    for index, (name, limits, color) in enumerate(OPTIC_SERIES[: optics.shape[1]]):
        optic = optics[:, index]
        served = optic_index == index
        if served.sum() == 0:
            axes.plot(
                *optic,
                "o",
                color=color,
                ms=12,
                mec="0.4",
                mew=1.0,
                zorder=8,
                label=f"{name} (idle)",
            )
            continue
        theta = np.degrees(
            geometry_utils.scattering_angles(volumes[:, served], optic, direction)
        )
        low, high = limits
        valid = bool(theta.min() >= low and theta.max() <= high)
        # The lens sits at the design standoff, outside the cross-section, so the marker
        # is the port: where the line of sight from the served volumes crosses the
        # design surface.
        centroid = volumes[:, served].mean(axis=1)
        standoff = float(np.linalg.norm(optic - centroid))
        crossing = geometry_utils.segment_curve_span(surface_curve, centroid, optic)
        anchor = (
            centroid + crossing[1] * (optic - centroid)
            if crossing is not None
            else optic
        )

        if show_admissible:
            allowed = geometry_utils.admissible_surface_angles(
                volumes[:, served], surface, direction, limits, standoff
            )
            if allowed.size:
                axes.plot(
                    *np.asarray(
                        geometry_utils.surface_point_at_angle(surface, allowed)
                    ),
                    ".",
                    color=color,
                    ms=4,
                    alpha=0.55,
                    zorder=2,
                )
        if show_sight_lines:
            for volume in np.flatnonzero(served):
                axes.plot(
                    [volumes[0, volume], anchor[0]],
                    [volumes[1, volume], anchor[1]],
                    "-",
                    color=color,
                    lw=0.6,
                    alpha=0.30,
                    zorder=1,
                )
        side = "LFS" if optic[0] > axis[0] else "HFS"
        axes.plot(
            *anchor,
            "o",
            color=color,
            ms=12,
            mec="k" if valid else "red",
            mew=1.0 if valid else 2.5,
            zorder=8,
            label=f"{name} port ({side}), lens {standoff:.2f} m behind",
        )
        # The bars are often sub-pixel, hence the relative size is spelled out in the
        # legend rather than left to be read off the plot.
        lit_set = light[served] > 0
        relative = (
            sigma[served][lit_set] / light[served][lit_set]
            if lit_set.any()
            else np.array([np.nan])
        )
        axes_light.errorbar(
            along[served],
            np.where(lit_set, light[served], np.nan),
            yerr=np.where(lit_set, sigma[served], np.nan),
            fmt="o-",
            color=color,
            ms=5,
            lw=1.2,
            capsize=3,
            label=(
                f"{name} ({int(lit_set.sum())}/{int(served.sum())} lit, "
                f"sigma/L {100 * np.nanmin(relative):.3f}-"
                f"{100 * np.nanmax(relative):.3f}%)"
            ),
        )

    if span is not None:
        edge_points = (
            beam_start[:, None]
            + np.asarray(span)[None, :] * (beam_end - beam_start)[:, None]
        )
        axes.plot(*edge_points, "|", color="k", ms=9, mew=1.0, zorder=5)
    lit = light > 0.0
    if lit.any():
        scatter = axes.scatter(
            *volumes[:, lit],
            c=light[lit],
            s=60,
            cmap="plasma",
            norm=matplotlib.colors.LogNorm(
                vmin=light[lit].min(), vmax=light[lit].max()
            ),
            ec="k",
            lw=0.6,
            zorder=6,
        )
        axes.get_figure().colorbar(
            scatter, ax=axes, label=r"collected light $\sum_i D_i$", shrink=0.8
        )
    if (~lit).any():
        axes.scatter(
            *volumes[:, ~lit],
            facecolors="none",
            s=60,
            ec="0.5",
            lw=0.8,
            zorder=6,
            label="no light (outside plasma)",
        )
    # A volume can carry light and still be absent from the likelihood, which is how a
    # diagnostic-information sweep switches volumes off without moving any of them.
    if active_volume_indices is not None:
        masked = np.ones(n_elements, dtype=bool)
        masked[list(active_volume_indices)] = False
        if masked.any():
            axes.scatter(
                *volumes[:, masked],
                marker="x",
                s=70,
                c="0.2",
                lw=1.4,
                zorder=7,
                label=f"masked ({int(masked.sum())} of {n_elements} not in likelihood)",
            )
            axes_light.plot(
                along[masked],
                np.where(lit[masked], light[masked], np.nan),
                "x",
                color="0.2",
                ms=9,
                mew=1.6,
                zorder=7,
                label=f"masked ({int(masked.sum())} of {n_elements} not in likelihood)",
            )
    axes.plot(*axis, "+", color="0.45", ms=15, mew=2.0, zorder=9, label="magnetic axis")

    axes.set_aspect("equal")
    axes.grid(alpha=0.25)
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    # Legend below the axes so it never covers the cross-section.
    axes.legend(
        fontsize=7.5,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        frameon=False,
        borderaxespad=0.0,
    )

    if span is not None:
        for index, crossing in enumerate(span):
            axes_light.axvline(
                direction @ (beam_start + crossing * (beam_end - beam_start) - centre),
                color="k",
                ls="--",
                lw=1.2,
                label="LCFS crossing" if index == 0 else None,
            )
    axes_light.axvline(
        direction @ (axis - centre), color="0.45", ls=":", lw=1.6, label="magnetic axis"
    )
    axes_light.set_yscale("log")
    axes_light.set_xlabel("distance along chord from centre [m]")
    axes_light.set_ylabel(r"collected light $\sum_i D_i$ (photoelectrons)")
    axes_light.grid(alpha=0.25, which="both")
    axes_light.legend(fontsize=8)
    # The panels share one figure in the default case and sit in separate figures when
    # the caller supplies its own axes to save them individually, so the shared figure
    # must not be laid out twice.
    left_figure, right_figure = axes.get_figure(), axes_light.get_figure()
    left_figure.tight_layout()
    if right_figure is not left_figure:
        right_figure.tight_layout()
    return left_figure, right_figure


def _poloidal(
    positions: jt.Float[NpOrJaxArray, "n_point 3"],
    toroidal_angle: float,
) -> jt.Float[np.ndarray, "2 n_point"]:
    """Cartesian points as signed `(R, Z)` columns in the plane at `toroidal_angle`.

    The projection onto the plane's radial direction keeps the sign, which `hypot` would
    lose for a lens standing beyond the machine axis of a unit-scale equilibrium.
    """
    cartesian = np.asarray(positions)
    radial = cartesian[:, 0] * np.cos(toroidal_angle) + cartesian[:, 1] * np.sin(
        toroidal_angle
    )
    return np.stack([radial, cartesian[:, 2]])


def plot_bands_on_spectrum(
    geometry: ts_types.ThomsonScatteringGeometry,
    response: ts_types.PolychromatorResponse,
    expected_temperature_ev: jt.Float[NpOrJaxArray, " n_volume"],
    volume_indices: tuple[int, ...] | None = None,
    n_columns: int = 3,
    titles: tuple[str, ...] | None = None,
) -> mpl_figure.Figure:
    """Polychromator passbands over the normalized spectrum, one panel per volume.

    A panel is one scattering volume at its own angle and expected temperature, its
    bands drawn at the scale `response.band_scales` carries for it.

    Args:
        geometry: The layout, fixing every volume's scattering angle.
        response: The response the placement carries, its `band_scales` set by the
            design point's rule; unscaled bands are drawn where it carries none.
        expected_temperature_ev: Expected electron temperature at every volume in eV.
        volume_indices: Volumes to draw; None draws every volume.
        n_columns: Panels per row.
        titles: One panel title per drawn volume; None names the volume, its angle
            and its temperature.

    Returns:
        The figure.
    """
    laser_m = float(response.laser_wavelength)
    wavelength_nm = jnp.linspace(600.0, 1100.0, 2001)
    wavelength = wavelength_nm * 1e-9
    angles_deg = np.degrees(np.asarray(ts.volume_scattering_angles(geometry)))
    temperatures = np.asarray(expected_temperature_ev, dtype=float)
    scales = _volume_band_scales(response, angles_deg.shape[0])
    indices = (
        tuple(range(angles_deg.shape[0]))
        if volume_indices is None
        else tuple(volume_indices)
    )
    n_rows = math.ceil(len(indices) / n_columns)
    figure, axes = pubfig.subplots(
        n_rows, n_columns, width="double", aspect=0.75, sharex=True
    )
    axes = np.asarray(axes).reshape(n_rows, n_columns)
    n_channel = np.asarray(response.filter_centers).shape[0]
    for position, volume in enumerate(indices):
        ax = axes[position // n_columns, position % n_columns]
        spectrum = ts.spectral_density_relativistic(
            wavelength, jnp.radians(angles_deg[volume]), temperatures[volume], laser_m
        )
        spectrum = spectrum / jnp.max(spectrum)
        low_edges, high_edges = ts.band_edges(response, jnp.array([scales[volume]]))
        bands = list(zip(low_edges[0].tolist(), high_edges[0].tolist()))
        for channel, (low, high) in enumerate(bands):
            ax.axvspan(
                low * 1e9,
                high * 1e9,
                alpha=0.35,
                lw=0.0,
                color=pubfig.CATEGORICAL_PALETTE[channel],
            )
        ax.plot(wavelength_nm, spectrum, "-", lw=1.0, color="k")
        ax.axvline(laser_m * 1e9, color="0.25", lw=1.2, ls=(0, (4, 2)))
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks([600.0, 700.0, 800.0, 900.0, 1000.0])
        # Five four-digit labels do not fit a third of a double column at the house
        # tick size, so this figure alone uses smaller x tick labels.
        ax.tick_params(axis="x", labelsize=plt.rcParams["xtick.labelsize"] - 2.0)
        ax.set_title(
            titles[position]
            if titles is not None
            else rf"vol. {volume}, ${angles_deg[volume]:.0f}^\circ$, "
            rf"${temperatures[volume]:.0f}$ eV",
            fontsize="small",
        )
        ax.text(
            0.03,
            0.92,
            rf"$s = {scales[volume]:.2f}$",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
        if position // n_columns == n_rows - 1:
            ax.set_xlabel(r"$\lambda$ (nm)")
        if position % n_columns == 0:
            ax.set_ylabel(r"$S / S_{\mathrm{max}}$")
    for position in range(len(indices), n_rows * n_columns):
        axes[position // n_columns, position % n_columns].set_axis_off()
    handles = [
        matplotlib.patches.Patch(
            color=pubfig.CATEGORICAL_PALETTE[channel],
            alpha=0.35,
            label=f"CH{channel + 1}",
        )
        for channel in range(n_channel)
    ]
    handles.append(
        matplotlib.lines.Line2D(
            [],
            [],
            color="0.25",
            lw=1.2,
            ls=(0, (4, 2)),
            label=rf"$\lambda_0 = {laser_m * 1e9:.0f}$ nm",
        )
    )
    # The panel titles are wider than the panels, so an "outside right" legend, whose
    # reserved width is its own, gets run into by the rightmost title. Shrinking the
    # layout rect instead keeps a strip of the figure free for the legend alone.
    figure.legend(
        handles=handles,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),
        bbox_transform=figure.transFigure,
    )
    figure.get_layout_engine().set(rect=(0.0, 0.0, 0.84, 1.0))
    return figure


def _volume_band_scales(
    response: ts_types.PolychromatorResponse, n_volume: int
) -> jt.Float[np.ndarray, " n_volume"]:
    """The scale every volume's bands are drawn at, ones when the response has none."""
    if response.band_scales is None:
        return np.ones(n_volume)
    return np.asarray(response.band_scales, dtype=float)


def draw_interferometer_layout(
    axes: mpl_axes.Axes,
    coefficients: desc_types.DescFourierZernikeCoefficients,
    chords: list[if_types.InterferometerGeometry],
    design: diagnostic_design_point.DiagnosticDesignPoint,
    active_chord_indices: tuple[int, ...] | None = None,
) -> None:
    """Draw one interferometer layout on `axes`: design surface, boundary, chords,
    impacts, axis.

    The chords are drawn as the pipeline built them, their endpoints read back from the
    lab-frame geometries and projected into the poloidal plane. Only the frame and the
    axis carry legend labels, so panels of one figure can share a legend.

    Args:
        axes: The axes to draw on.
        coefficients: DESC geometry coefficients of the equilibrium.
        chords: Interferometer lines of sight of the layout.
        design: Design point supplying the plane and the surface offset.
        active_chord_indices: Chords a likelihood actually reads; every other one is
            drawn dashed and grey. None marks none, which is the layout as built.
    """
    toroidal_angle = np.radians(design.toroidal_angle_deg)
    plasma, surface = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, design.surface_offset
    )
    surface_curve, centre = np.asarray(surface.curve), np.asarray(surface.centre)
    axis = geometry_utils.magnetic_axis(coefficients, toroidal_angle)
    starts_rz, ends_rz = _chord_endpoints_rz(chords)
    # Each chord's own perpendicular from the centre, which is the shared fan normal
    # scaled by the impact parameter when the chords are parallel and still the right
    # annotation when they are not.
    directions = ends_rz - starts_rz
    directions = directions / np.linalg.norm(directions, axis=0)
    feet = starts_rz + directions * np.einsum(
        "ij,ij->j", centre[:, None] - starts_rz, directions
    )
    axes.plot(
        *surface_curve,
        "--",
        color="0.45",
        lw=0.8,
        label=rf"auxiliary design surface ($+{design.surface_offset:.2f}$ m)",
    )
    axes.plot(*plasma, "-", color="k", lw=1.0, label=r"plasma boundary ($\rho = 1$)")
    masked_chords = (
        np.zeros(len(chords), dtype=bool)
        if active_chord_indices is None
        else ~np.isin(np.arange(len(chords)), list(active_chord_indices))
    )
    for index in range(len(chords)):
        color = pubfig.CATEGORICAL_PALETTE[index % len(pubfig.CATEGORICAL_PALETTE)]
        axes.plot(
            [starts_rz[0, index], ends_rz[0, index]],
            [starts_rz[1, index], ends_rz[1, index]],
            ls=":" if masked_chords[index] else "-",
            color="0.7" if masked_chords[index] else color,
            lw=1.0 if masked_chords[index] else 1.2,
            zorder=3,
        )
        axes.plot(
            *np.stack([centre, feet[:, index]], axis=1),
            ":",
            color=color,
            lw=0.7,
            zorder=1,
        )
    axes.plot(*axis, "+", color="k", ms=6, mew=1.0, zorder=4, label="magnetic axis")
    if masked_chords.any():
        axes.plot(
            [],
            [],
            ":",
            color="0.7",
            lw=1.0,
            label=f"masked ({int(masked_chords.sum())} of {len(chords)})",
        )
    axes.set_aspect("equal")
    axes.set_xlabel(r"$R$ (m)")
    axes.set_ylabel(r"$Z$ (m)")
    margin = 0.05 * float(np.ptp(surface_curve[0]))
    axes.set_xlim(
        float(surface_curve[0].min()) - margin, float(surface_curve[0].max()) + margin
    )
    axes.set_ylim(
        float(surface_curve[1].min()) - margin, float(surface_curve[1].max()) + margin
    )


def draw_interferometer_line_integrals(
    axes: mpl_axes.Axes,
    coefficients: desc_types.DescFourierZernikeCoefficients,
    chords: list[if_types.InterferometerGeometry],
    electron_density_fn: Callable[[NpOrJaxArray], NpOrJaxArray],
    design: diagnostic_design_point.DiagnosticDesignPoint,
    n_sample: int = 128,
) -> None:
    """Draw one fan's line-integrated densities on `axes`, with the instrument's sigma.

    Against the chord index, not the impact parameter, which is not an ordering once the
    chords are free to cross. The colour of each point is the chord's colour in
    `draw_interferometer_layout`.

    Args:
        axes: The axes to draw on.
        coefficients: DESC geometry coefficients of the equilibrium.
        chords: Interferometer lines of sight of the layout.
        electron_density_fn: `ne(rho)` profile of the plotted state.
        design: Design point supplying the plane and the surface offset.
        n_sample: Quadrature points per chord of the line integral.
    """
    toroidal_angle = np.radians(design.toroidal_angle_deg)
    plasma, _ = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, design.surface_offset
    )
    starts_rz, ends_rz = _chord_endpoints_rz(chords)
    densities = np.array(
        [
            geometry_utils.chord_line_integrated_density(
                coefficients,
                toroidal_angle,
                plasma,
                starts_rz[:, index],
                ends_rz[:, index],
                electron_density_fn,
                n_sample=n_sample,
            )
            for index in range(len(chords))
        ]
    )
    # The instrument's own sigma, from the one place it is defined, so the error bars
    # here are the ones the likelihood uses: one constant, the same on every chord.
    sigmas = np.asarray(if_utils.compute_interferometer_uncertainty(densities))
    index_axis = np.arange(1, len(chords) + 1)
    axes.errorbar(
        index_axis,
        densities,
        yerr=sigmas,
        fmt="-",
        color="0.4",
        lw=0.8,
        capsize=2.0,
        zorder=2,
        label=(
            rf"$N \pm \sigma$ "
            rf"({if_utils.LINE_DENSITY_NOISE_M2:.1e}"
            rf"$\,$m$^{{-2}}$)"
        ),
    )
    for index in range(len(chords)):
        axes.plot(
            index_axis[index],
            densities[index],
            "o",
            ms=5.0,
            zorder=3,
            color=pubfig.CATEGORICAL_PALETTE[index % len(pubfig.CATEGORICAL_PALETTE)],
        )
    axes.set_xticks(index_axis)
    axes.set_xlabel("line of sight")
    axes.set_ylabel(r"$N$ (m$^{-2}$)")
    axes.margins(x=0.12, y=0.12)


def plot_interferometer_layout(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    chords: list[if_types.InterferometerGeometry],
    electron_density_fn: Callable[[NpOrJaxArray], NpOrJaxArray],
    design: diagnostic_design_point.DiagnosticDesignPoint,
    n_sample: int = 128,
    active_chord_indices: tuple[int, ...] | None = None,
) -> tuple[mpl_figure.Figure, mpl_figure.Figure]:
    """Interferometer geometry and its line-integrated densities, as two figures.

    One colour per line of sight is shared by both figures, which is what identifies a
    chord across them; the exact impact parameters are in `interferometer_chord_table`.

    Args:
        coefficients: DESC geometry coefficients of the equilibrium.
        chords: Interferometer lines of sight of the layout.
        electron_density_fn: `ne(rho)` profile of the plotted state.
        design: Design point supplying the plane, the surface offset and the laser.
        n_sample: Quadrature points per chord of the line integral.
        active_chord_indices: Chords a likelihood actually reads; every other one is
            drawn dashed and grey. None marks none, which is the layout as built.

    Returns:
        The chord-layout figure and the line-integrated-density figure.
    """
    layout_figure, layout_axes = pubfig.subplots(1, 1, width="single", aspect=1.0)
    draw_interferometer_layout(
        layout_axes.flat[0], coefficients, chords, design, active_chord_indices
    )
    # A figure-level legend under the axes: the chords fill the frame, and unlike an
    # anchored axes legend constrained_layout reserves the room for this one. The chords
    # themselves are not in it; their colour is what links them to the density figure.
    layout_figure.legend(
        handles=layout_axes.flat[0].get_legend_handles_labels()[0],
        ncol=2,
        loc="outside lower center",
    )
    density_figure, density_axes = pubfig.subplots(1, 1, width="single", aspect=0.85)
    draw_interferometer_line_integrals(
        density_axes.flat[0],
        coefficients,
        chords,
        electron_density_fn,
        design,
        n_sample,
    )
    density_axes.flat[0].legend(loc="lower center")
    return layout_figure, density_figure


def _chord_endpoints_rz(
    chords: list[if_types.InterferometerGeometry],
) -> tuple[jt.Float[np.ndarray, "2 n_chord"], jt.Float[np.ndarray, "2 n_chord"]]:
    """Lab-frame chord endpoints projected into the poloidal plane, `(R, Z)` columns."""
    # Reshaped so a layout without chords still yields two `(2, 0)` arrays.
    starts = np.asarray(
        [np.asarray(chord.chord_start) for chord in chords], dtype=float
    ).reshape(-1, 3)
    ends = np.asarray(
        [np.asarray(chord.chord_end) for chord in chords], dtype=float
    ).reshape(-1, 3)
    return (
        np.stack([np.hypot(starts[:, 0], starts[:, 1]), starts[:, 2]]),
        np.stack([np.hypot(ends[:, 0], ends[:, 1]), ends[:, 2]]),
    )
