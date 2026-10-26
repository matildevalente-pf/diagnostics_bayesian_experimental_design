import numbers

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib import axes as mpl_axes
from matplotlib import figure as mpl_figure
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)

_RHO_CURVE_RESOLUTION = 256
"""Poloidal resolution of each constant-`rho` surface curve."""

_THETA_CURVE_RESOLUTION = 101
"""Radial resolution of each constant-`theta` contour."""


def plot_surfaces(
    eq: desc_types.DescEquilibrium,
    rho: int | NpOrJaxArray = 8,
    theta: int | NpOrJaxArray = 8,
    phi: float | int | NpOrJaxArray | None = None,
    ax: mpl_axes.Axes | np.ndarray | None = None,
    return_data: bool = False,
    **kwargs,
) -> tuple[mpl_figure.Figure, np.ndarray] | tuple[mpl_figure.Figure, np.ndarray, dict]:
    """Plot flux surfaces of an equilibrium, in the style of `desc.plotting.plot_surfaces`.

    Each panel is one toroidal cross-section with constant-`rho` surfaces, the last
    closed flux surface emphasized, constant-`theta` contours (DESC's native poloidal
    angle, not the straight-field-line `vartheta`), and the magnetic axis.

    Args:
        eq: Equilibrium to plot.
        rho: Values of `rho` to plot contours of; an integer plots that many contours
            linearly spaced in `(0, 1]`.
        theta: Values of `theta` to plot contours of; an integer plots that many
            contours linearly spaced in `[0, 2pi)`.
        phi: Values of `phi` to plot cross-sections at; an integer plots that many
            sections linearly spaced over one field period. Default is 1 section for
            axisymmetric equilibria and 6 otherwise.
        ax: Axis (or array of axes, one per `phi`) to plot on; a fresh figure if None.
        return_data: Also return the plotted curves.
        **kwargs: Figure and style options: `figsize`, `legend`, `legend_kw`,
            `title_fontsize`, `xlabel_fontsize`, `ylabel_fontsize`, `label`,
            `rho_color`/`rho_ls`/`rho_lw`, `theta_color`/`theta_ls`/`theta_lw`,
            `lcfs_color`/`lcfs_ls`/`lcfs_lw`, `axis_color`/`axis_alpha`/`axis_marker`/
            `axis_size`.

    Returns:
        The figure and the axes array, plus (with `return_data`) a dict with keys
        `rho_R_coords`, `rho_Z_coords`, `theta_R_coords`, `theta_Z_coords`, each a
        list over `phi` of `(n_curve, n_point)` arrays.
    """
    figsize = kwargs.pop("figsize", None)
    legend = kwargs.pop("legend", True)
    legend_kw = kwargs.pop("legend_kw", {})
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    label = kwargs.pop("label", None)
    rho_color = kwargs.pop("rho_color", "tab:blue")
    rho_ls = kwargs.pop("rho_ls", "-")
    rho_lw = kwargs.pop("rho_lw", 1.0)
    theta_color = kwargs.pop("theta_color", "tab:red")
    theta_ls = kwargs.pop("theta_ls", ":")
    theta_lw = kwargs.pop("theta_lw", 1.0)
    lcfs_color = kwargs.pop("lcfs_color", rho_color)
    lcfs_ls = kwargs.pop("lcfs_ls", "-")
    lcfs_lw = kwargs.pop("lcfs_lw", 1.8)
    axis_color = kwargs.pop("axis_color", "k")
    axis_alpha = kwargs.pop("axis_alpha", 1.0)
    axis_marker = kwargs.pop("axis_marker", "o")
    axis_size = kwargs.pop("axis_size", 36)
    assert len(kwargs) == 0, (
        f"plot_surfaces got unexpected keyword argument: {kwargs.keys()}"
    )

    coefficients = eq.coefficients
    if isinstance(rho, numbers.Integral):
        rho = np.linspace(0.0, 1.0, rho + 1)[1:]
    rho = np.atleast_1d(np.asarray(rho))
    if isinstance(theta, numbers.Integral):
        theta = np.linspace(0.0, 2.0 * np.pi, theta, endpoint=False)
    theta = np.atleast_1d(np.asarray(theta))
    phi = _parse_phi(phi, coefficients)
    n_phi = len(phi)

    n_rows = int(np.floor(np.sqrt(n_phi)))
    n_columns = int(np.ceil(n_phi / n_rows))
    if ax is None:
        if figsize is None:
            figsize = (4 * n_columns, 5 * n_rows)
        figure, ax = plt.subplots(
            n_rows, n_columns, figsize=figsize, constrained_layout=True, squeeze=False
        )
    ax = np.atleast_1d(np.asarray(ax)).ravel()
    figure = ax[0].get_figure()
    for panel in ax[n_phi:]:
        panel.set_visible(False)

    plot_data: dict = {
        "rho_R_coords": [],
        "rho_Z_coords": [],
        "theta_R_coords": [],
        "theta_Z_coords": [],
    }
    poloidal_grid = np.linspace(0.0, 2.0 * np.pi, _RHO_CURVE_RESOLUTION)
    radial_grid = np.linspace(0.0, 1.0, _THETA_CURVE_RESOLUTION)
    for index, (panel, phi_value) in enumerate(zip(ax, phi)):
        theta_r, theta_z = [], []
        for theta_value in theta:
            r, z = desc_utils.evaluate_rz(
                coefficients,
                radial_grid,
                np.full(radial_grid.shape, theta_value),
                np.full(radial_grid.shape, phi_value),
            )
            r, z = np.asarray(r), np.asarray(z)
            panel.plot(r, z, color=theta_color, ls=theta_ls, lw=theta_lw)
            theta_r.append(r)
            theta_z.append(z)

        rho_r, rho_z = [], []
        for rho_value in rho:
            r, z = desc_utils.evaluate_rz(
                coefficients,
                np.full(poloidal_grid.shape, rho_value),
                poloidal_grid,
                np.full(poloidal_grid.shape, phi_value),
            )
            r, z = np.asarray(r), np.asarray(z)
            is_lcfs = rho_value == rho.max()
            panel.plot(
                r,
                z,
                color=lcfs_color if is_lcfs else rho_color,
                ls=lcfs_ls if is_lcfs else rho_ls,
                lw=lcfs_lw if is_lcfs else rho_lw,
                label=label if is_lcfs and index == 0 else None,
            )
            rho_r.append(r)
            rho_z.append(z)

        axis_r, axis_z = desc_utils.evaluate_rz(
            coefficients, np.zeros(1), np.zeros(1), np.array([phi_value])
        )
        panel.scatter(
            np.asarray(axis_r),
            np.asarray(axis_z),
            color=axis_color,
            alpha=axis_alpha,
            marker=axis_marker,
            s=axis_size,
            zorder=5,
        )

        panel.set_aspect("equal")
        panel.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=4, min_n_ticks=3)
        )
        panel.set_xlabel(r"$R$ (m)", fontsize=xlabel_fontsize)
        panel.set_ylabel(r"$Z$ (m)", fontsize=ylabel_fontsize)
        panel.set_title(
            rf"$\phi = {phi_value / np.pi:.2f}\pi$" if phi_value else r"$\phi = 0$",
            fontsize=title_fontsize,
        )
        plot_data["rho_R_coords"].append(np.stack(rho_r))
        plot_data["rho_Z_coords"].append(np.stack(rho_z))
        # `theta` may be empty (plot_comparison hides the theta contours).
        plot_data["theta_R_coords"].append(
            np.stack(theta_r) if theta_r else np.empty((0, _THETA_CURVE_RESOLUTION))
        )
        plot_data["theta_Z_coords"].append(
            np.stack(theta_z) if theta_z else np.empty((0, _THETA_CURVE_RESOLUTION))
        )

    if label is not None and legend:
        ax[0].legend(**legend_kw)

    if return_data:
        return figure, ax, plot_data
    return figure, ax


def plot_comparison(
    eqs: list[desc_types.DescEquilibrium],
    rho: int | NpOrJaxArray = 8,
    theta: int | NpOrJaxArray = 8,
    phi: float | int | NpOrJaxArray | None = None,
    ax: mpl_axes.Axes | np.ndarray | None = None,
    cmap: str = "rainbow",
    color: list | None = None,
    lw: list | None = None,
    ls: list | None = None,
    labels: list[str] | None = None,
    return_data: bool = False,
    **kwargs,
) -> tuple[mpl_figure.Figure, np.ndarray] | tuple[mpl_figure.Figure, np.ndarray, dict]:
    """Plot comparison between flux surfaces of multiple equilibria, in the style of
    `desc.plotting.plot_comparison`.

    Overlays `plot_surfaces` of every equilibrium in one figure, one colour per
    equilibrium (theta contours are hidden, so the panels stay readable). If the
    equilibria have differing field periods, `phi` must be given explicitly.

    Args:
        eqs: Equilibria to compare.
        rho: Values of `rho` to plot contours of; an integer plots that many contours
            linearly spaced in `(0, 1]`.
        theta: Unused (kept for signature parity with desc; theta contours are hidden
            in comparisons).
        phi: Values of `phi` to plot cross-sections at; an integer plots that many
            sections linearly spaced over one field period. Default is 1 section for
            axisymmetric equilibria and 6 otherwise.
        ax: Axis (or array of axes, one per `phi`) to plot on; a fresh figure if None.
        cmap: Colormap discretized into `len(eqs)` colours.
        color: One colour per equilibrium; overrides `cmap`.
        lw: One line width per equilibrium.
        ls: One line style per equilibrium.
        labels: One legend label per equilibrium.
        return_data: Also return the plotted curves.
        **kwargs: Figure options as in `plot_surfaces` (`figsize`, `legend`,
            `legend_kw`, `title_fontsize`, `xlabel_fontsize`, `ylabel_fontsize`).

    Returns:
        The figure and the axes array, plus (with `return_data`) a dict with the
        `plot_surfaces` data keys, each a list over equilibria.
    """
    if phi is None and not all(
        eq.n_field_periods == eqs[0].n_field_periods for eq in eqs
    ):
        raise ValueError(
            "supplied equilibria must have the same number of field periods, or "
            "`phi` must be given explicitly; got NFPs "
            f"{[eq.n_field_periods for eq in eqs]}."
        )
    figsize = kwargs.pop("figsize", None)
    legend = kwargs.pop("legend", True)
    legend_kw = kwargs.pop("legend_kw", {})
    title_fontsize = kwargs.pop("title_fontsize", None)
    xlabel_fontsize = kwargs.pop("xlabel_fontsize", None)
    ylabel_fontsize = kwargs.pop("ylabel_fontsize", None)
    assert len(kwargs) == 0, (
        f"plot_comparison got unexpected keyword argument: {kwargs.keys()}"
    )

    n_eqs = len(eqs)
    if color is None:
        color = plt.get_cmap(cmap, n_eqs)(np.linspace(0.0, 1.0, n_eqs))
    if lw is None:
        lw = [1.0] * n_eqs
    if ls is None:
        ls = ["-"] * n_eqs
    if labels is None:
        labels = [str(i) for i in range(n_eqs)]

    phi = _parse_phi(phi, eqs[0].coefficients)
    n_phi = len(phi)
    n_rows = int(np.floor(np.sqrt(n_phi)))
    n_columns = int(np.ceil(n_phi / n_rows))
    if ax is None:
        if figsize is None:
            figsize = (4 * n_columns, 5 * n_rows)
        figure, ax = plt.subplots(
            n_rows, n_columns, figsize=figsize, constrained_layout=True, squeeze=False
        )
    ax = np.atleast_1d(np.asarray(ax)).ravel()
    figure = ax[0].get_figure()

    plot_data: dict = {
        "rho_R_coords": [],
        "rho_Z_coords": [],
        "theta_R_coords": [],
        "theta_Z_coords": [],
    }
    for i, eq in enumerate(eqs):
        _, _, eq_data = plot_surfaces(
            eq,
            rho=rho,
            theta=np.array([]),
            phi=phi,
            ax=ax,
            rho_color=color[i % len(color)],
            rho_ls=ls[i % len(ls)],
            rho_lw=lw[i % len(lw)],
            lcfs_color=color[i % len(color)],
            lcfs_ls=ls[i % len(ls)],
            lcfs_lw=lw[i % len(lw)],
            axis_color=color[i % len(color)],
            axis_alpha=0.0,
            label=labels[i % len(labels)],
            legend=False,
            title_fontsize=title_fontsize,
            xlabel_fontsize=xlabel_fontsize,
            ylabel_fontsize=ylabel_fontsize,
            return_data=True,
        )
        for key in plot_data:
            plot_data[key].append(eq_data[key])

    if any(labels) and legend:
        figure.legend(**legend_kw)

    if return_data:
        return figure, ax, plot_data
    return figure, ax


def _parse_phi(
    phi: float | int | NpOrJaxArray | None,
    coefficients: desc_types.DescFourierZernikeCoefficients,
) -> np.ndarray:
    """Resolve the `phi` argument to an array of cross-section angles."""
    if phi is None:
        axisymmetric = bool(
            (np.asarray(coefficients.r_modes)[:, 2] == 0).all()
            and (np.asarray(coefficients.z_modes)[:, 2] == 0).all()
        )
        phi = 1 if axisymmetric else 6
    if isinstance(phi, numbers.Integral):
        phi = np.linspace(
            0.0, 2.0 * np.pi / coefficients.n_field_periods, phi, endpoint=False
        )
    return np.atleast_1d(np.asarray(phi, dtype=float))
