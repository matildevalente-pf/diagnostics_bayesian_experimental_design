# Publication-figure style for matplotlib (LaTeX serif, boxed axes, print sizing).
# Typical use in a notebook:
#
#     pubfig.apply_style()
#     fig, axes = pubfig.subplots(1, 2, width="double", aspect=0.75)
#     ...
#     pubfig.save(fig, "figures/my_figure")  # writes a 300 dpi .pdf
#
# The module deliberately exposes only sizing, style and save helpers: the plotting
# itself stays in the caller so figures remain readable as ordinary matplotlib code.

import logging
import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure as mpl_figure

SINGLE_COLUMN_IN = 3.5
"""Single-column width: 8.9 cm."""

DOUBLE_COLUMN_IN = 7.2
"""Double-column width: 18.3 cm."""

DPI = 300
"""Minimum raster resolution for print."""

# Okabe-Ito, colour-blind safe. Ordered so the first pairs stay distinguishable
# in greyscale and avoid a red/green contrast.
PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

# Project-wide categorical palette for discrete series (polychromator channels, bars,
# grouped categories). Hues are spread further apart than `PALETTE`, so many series
# stay separable side by side, including when drawn as light shaded patches.
CATEGORICAL_PALETTE = (
    "#1D67A9",  # blue
    "#CE5F2C",  # orange/red
    "#C46F9D",  # pink
    "#1A916B",  # green
    "#DF9A31",  # ochre/orange
    "#8C564B",  # brown
    "#9467BD",  # purple
    "#17BECF",  # cyan
)

# Computer Modern Roman, so figure text matches LaTeX body text without needing a
# usetex round trip (the container has no dvipng, which usetex+Agg requires).
_SERIF_STACK = ("cmr10", "CMU Serif", "TeX Gyre Termes", "DejaVu Serif")


def apply_style(base_font_pt: float = 12.0) -> None:
    """Set rcParams to the house print style.

    LaTeX-style serif text, a full box around the axes, ticks pointing inward on
    all four sides.

    Args:
        base_font_pt: Axis-label size at final print size. Ticks are set one
            point smaller. Journals that mandate 5 to 7 pt text (Nature) need an
            explicit `apply_style(base_font_pt=7)`.
    """
    tick_pt = max(base_font_pt - 1.0, 5.0)
    # Embedding cmr10 makes fontTools log a "timestamp seems very low" notice on
    # every PDF write. Harmless, but it buries real warnings in a notebook.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            "pdf.fonttype": 42,  # embed TrueType, keep text selectable/editable
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "serif",
            "font.serif": list(_SERIF_STACK),
            "mathtext.fontset": "cm",
            # cmr10 has no U+2212, so numeric labels must go through mathtext or
            # matplotlib warns and falls back on every negative tick.
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": base_font_pt,
            "axes.labelsize": base_font_pt,
            "axes.titlesize": base_font_pt,
            "xtick.labelsize": tick_pt,
            "ytick.labelsize": tick_pt,
            "legend.fontsize": tick_pt,
            "legend.title_fontsize": base_font_pt,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "patch.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderpad": 0.2,
            "legend.labelspacing": 0.3,
            "errorbar.capsize": 2.0,
            "axes.prop_cycle": plt.cycler(color=list(PALETTE)),
        }
    )


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    width: Literal["single", "double"] | float = "single",
    aspect: float = 0.75,
    **kwargs,
) -> tuple[mpl_figure.Figure, np.ndarray]:
    """Create a figure at exact column width.

    Args:
        width: `"single"` (3.5 in), `"double"` (7.2 in), or an explicit width in
            inches for a partial-width figure.
        aspect: Height / width of a single panel. The figure height is
            `nrows * aspect * (width / ncols)`, so panels stay square-ish
            whatever the grid.
    """
    width_in = {"single": SINGLE_COLUMN_IN, "double": DOUBLE_COLUMN_IN}.get(width, width)
    panel_width_in = width_in / ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_in, nrows * aspect * panel_width_in),
        constrained_layout=True,
        **kwargs,
    )
    return fig, np.atleast_1d(np.asarray(axes, dtype=object))


def save(fig: mpl_figure.Figure, path_without_extension: str) -> None:
    """Write the figure as a vector PDF.

    PDF only, deliberately. A parallel PNG doubles the artifacts in every results
    folder, is not gitignored so it lands in a commit, and goes stale the moment the PDF
    is regenerated without it. Anything that needs a raster (a slide, a preview) can
    render one from the PDF at whatever resolution it wants.

    Creates the parent directory if it does not exist.
    """
    path = pathlib.Path(path_without_extension)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
