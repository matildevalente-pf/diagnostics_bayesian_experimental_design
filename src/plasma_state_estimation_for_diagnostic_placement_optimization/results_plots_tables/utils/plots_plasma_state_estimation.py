# Figures of the plasma-state-estimation area: recoveries against the truth, the
# recipes' bands and metrics side by side, the parametrization and the process prior.

import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import axes as mpl_axes
from matplotlib import figure as mpl_figure
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_types,
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E501
    parametrization_fit_utils,
    pubfig,
)

_ABLATION_PROFILES = (
    ("ne", "electron_density", r"$n_e$ [m$^{-3}$]", r"$R_{n_e}$"),
    ("Te", "electron_temperature", r"$T_e$ [eV]", r"$R_{T_e}$"),
)
"""The two profiles the likelihood ablation reports: the name its tables column them
by, the `_LOCAL_ESTIMATE_FIELDS` name, and the axis labels of the profile and its
ratio."""


_LOCAL_ESTIMATE_FIELDS = {
    "electron_density": ("electron_density", "electron_density_sigma"),
    "electron_temperature": (
        "electron_temperature_ev",
        "electron_temperature_sigma_ev",
    ),
}
"""Value and width field of a `LocalStateEstimate` each profile panel draws.

The pressure panel draws none: a volume's spectrum fixes its `ne` and its `Te`, and a
pressure point would restate the two it already shows.
"""


ARM_LABELS = {
    "map_laplace": "MAP + Laplace",
    "mcmc_nuts": "NUTS from the MAP",
}
"""How each arm of the `mcmc_benchmark` study is named in a figure.

Two arms share the sampler and differ in where their chains start, so a label that named
only the recipe would put one name on two different runs.
"""


def plot_map_convergence(
    runs: list[
        tuple[
            inference_types.PlasmaStateInferenceResult,
            dict[str, np.ndarray],
            profile_parametrization_types.PlasmaProfiles,
        ]
    ],
    label: str | None = None,
) -> mpl_figure.Figure:
    """The mean and 5 to 95% quantiles over every MAP start of the runs of each profile
    parameter divided by its run's true value, over the MAP iterations, `ne` left and
    `Te` right, after Fig. 3 of Merlo et al. (2023, JINST 18 P11012), whose trials are
    the runs and whose columns are the other way round.

    Args:
        runs: Per run its stored record, `two_power_zero_edge` only, its arrays as
            `inference_store_utils.load_inference` returns them, and the ground
            truth's own profiles, projected onto the family.
        label: What the runs have in common, the legend title.

    Returns:
        The figure.

    Raises:
        ValueError: For any family other than `two_power_zero_edge`, or a record stored
            before the trajectory of every start was kept.
    """
    n_parameters = len(profile_parametrization_types.TWO_POWER_ZERO_EDGE_PARAM_NAMES)
    normalized_runs = []
    for result, arrays, truth_profiles in runs:
        family = result.settings.profile_family
        if family != "two_power_zero_edge":
            raise ValueError(f"no true parameter value to divide by for {family!r}")
        if "start_trajectories" not in arrays:
            raise ValueError(
                f"record {result.inference_id} keeps only its best start's trajectory; "
                "rerun it"
            )
        targets = dict(
            zip(
                ("ne", "te"),
                parametrization_fit_utils.target_parameters(family, truth_profiles),
            )
        )
        site_names = [str(name) for name in arrays["trajectory_site_names"]]
        start_trajectories = arrays["start_trajectories"]
        normalized_runs.append(
            np.stack(
                [
                    start_trajectories[:, :, site_names.index(f"{prefix}_{row}")]
                    / targets[prefix][row]
                    for prefix in ("ne", "te")
                    for row in range(n_parameters)
                ],
                axis=-1,
            )
        )
    n_iterations = max(run.shape[1] for run in normalized_runs)
    # A run that stopped early is held at its end point, as a start within a run is.
    normalized = np.concatenate(
        [
            np.pad(run, ((0, 0), (0, n_iterations - run.shape[1]), (0, 0)), mode="edge")
            for run in normalized_runs
        ]
    )
    steps = np.arange(n_iterations)
    color = pubfig.PALETTE[0]
    figure, axes_grid = pubfig.subplots(3, 2, width="double", aspect=0.62, sharex=True)
    for column, (prefix, profile_label) in enumerate(
        (("ne", r"$n_e$"), ("te", r"$T_e$"))
    ):
        for row, parameter in enumerate(
            profile_parametrization_types.TWO_POWER_ZERO_EDGE_PARAM_NAMES
        ):
            values = normalized[:, :, column * n_parameters + row]
            lower, upper = np.quantile(values, (0.05, 0.95), axis=0)
            panel = axes_grid[row, column]
            panel.plot(steps, values.mean(axis=0), color=color, lw=1.1, label="mean")
            panel.plot(
                steps, lower, color=color, ls="--", lw=0.8, label="5%-95% quantiles"
            )
            panel.plot(steps, upper, color=color, ls="--", lw=0.8)
            panel.axhline(1.0, color="tab:red", ls="--", lw=0.9, label="true")
            panel.set_ylabel(f"normalized {profile_label} {parameter}")
        axes_grid[-1, column].set_xlabel("MAP iteration")
    # On a translucent frame, so a start still away from the truth stays visible.
    axes_grid[0, 0].legend(
        loc="lower right",
        ncols=1,
        frameon=True,
        framealpha=0.7,
        edgecolor="none",
        title=_readable_label(label) if label else None,
    )
    return figure


def _draw_local_estimate(
    axes: mpl_axes.Axes,
    name: str,
    volume_rho: np.ndarray | None,
    estimate: thomson_scattering_utils.LocalStateEstimate,
) -> None:
    """Draw one panel's per-volume Thomson estimates, on the panels that have them.

    The points come from the same recorded signals the global inference was fitted to,
    each volume inverted on its own, so they show what one spectrum alone constrains
    rather than an independent measurement.
    """
    fields = _LOCAL_ESTIMATE_FIELDS.get(name)
    if fields is None:
        return
    assert volume_rho is not None, (
        "a local Thomson estimate needs the flux label of every volume; pass both of "
        "`synthetic_target_utils.thomson_local_estimates`"
    )
    axes.errorbar(
        np.asarray(volume_rho),
        np.asarray(getattr(estimate, fields[0])),
        yerr=np.asarray(getattr(estimate, fields[1])),
        fmt="o",
        ms=3.0,
        elinewidth=1.0,
        capsize=2.0,
        color="tab:red",
        zorder=5,
        label="TS inference per volume",
    )


def plot_prior_bands(
    profile_table: pd.DataFrame, prior_case: str = "prior"
) -> mpl_figure.Figure:
    """The prior band and median alone, `ne` then `Te`, every truth dashed over them.

    Args:
        profile_table: The long band table of `run_study.py`, every truth's rows.
        prior_case: Which `inference_case` holds the propagated prior draws.

    Returns:
        The figure.
    """
    figure, axes_array = pubfig.subplots(1, 2, width="double", aspect=0.8)
    prior = profile_table.loc[profile_table["inference_case"] == prior_case]
    # The prior does not depend on the truth, so any one truth's rows carry the band.
    band = prior.loc[prior["profile_id"] == prior["profile_id"].iloc[0]]
    for axes, (symbol, _, axis_label, _) in zip(axes_array, _ABLATION_PROFILES):
        _draw_prior_band(axes, band, symbol)
        for index, (_, truth) in enumerate(prior.groupby("profile_id", sort=False)):
            axes.plot(
                truth["rho"].to_numpy(),
                truth[f"{symbol}_truth"].to_numpy(),
                color="0.15",
                ls="--",
                lw=0.9,
                label="ground-truth profiles" if index == 0 else None,
                zorder=3,
            )
        axes.set_xlabel(r"$\rho$")
        axes.set_ylabel(axis_label)
    handles, labels = axes_array[0].get_legend_handles_labels()
    # Anchored to the right panel, not the figure: a figure legend tops out at the
    # figure edge, above the frame, and sits flush against the panel.
    axes_array[1].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.06, 1.0),
        borderaxespad=0.0,
        borderpad=0.0,
    )
    return figure


def plot_method_band_comparison(
    profile_table: pd.DataFrame,
    arms: tuple[str, ...],
    thomson_points: pd.DataFrame | None = None,
    labels: dict[str, str] | None = None,
    title: str | None = None,
    filled_bands: bool = False,
    initial_guess: bool = True,
) -> mpl_figure.Figure:
    """One truth's arms on two square panels, `ne` left and `Te` in keV right.

    Each arm draws its point estimate solid, the MAP or the sampled mean, and its 5-95%
    band dashed or shaded, in its own color over the truth; the start is dotted if arms
    share it.

    Args:
        profile_table: The long band table of `run_study.py`, one truth's rows only.
        arms: The `arm` values to draw, in the order they are listed.
        thomson_points: This truth's rows of the per-volume Thomson table, whose
            points every panel carries; None leaves them off.
        labels: How each arm is named in the legend; `ARM_LABELS` when None.
        title: What distinguishes this figure from its siblings, the legend title.
        filled_bands: Shade each band instead of dashing its two edges.
        initial_guess: Draw the shared start; False leaves it off.

    Returns:
        The figure.
    """
    arm_labels = labels or ARM_LABELS
    # The start is green and dotted; the arms take the other colors in order, the first
    # one blue, as the MAP is in `plot_map_convergence`.
    initial_color = pubfig.PALETTE[2]
    arm_colors = [color for color in pubfig.PALETTE if color != initial_color]
    blocks = {
        arm: profile_table.loc[profile_table["arm"] == arm]
        for arm in arms
        if (profile_table["arm"] == arm).any()
    }
    is_laplace = {
        arm: block["method"].iloc[0] == "map_laplace" for arm, block in blocks.items()
    }
    n_laplace = sum(is_laplace.values())
    # The tables carry temperatures in eV; the figure shows keV.
    scales = {"ne": 1.0, "Te": 1e-3}
    axis_labels = {"Te": r"$T_e$ [keV]"}
    points = (
        None
        if thomson_points is None
        else thomson_points.assign(
            Te=thomson_points["Te"] * scales["Te"],
            Te_sigma=thomson_points["Te_sigma"] * scales["Te"],
        )
    )
    figure, axes_array = pubfig.subplots(1, 2, width="double", aspect=1.0)
    for axes, (symbol, profile_name, axis_label, _) in zip(
        axes_array, _ABLATION_PROFILES
    ):
        scale = scales[symbol]
        first = next(iter(blocks.values()))
        rho = first["rho"].to_numpy()
        # The truth sits behind every estimate, which is what the eye compares to it.
        axes.plot(
            rho,
            first[f"{symbol}_truth"].to_numpy() * scale,
            color="k",
            lw=1.2,
            label="ground truth",
            zorder=1,
        )
        # The prior draw the reported MAP started from, drawn only when every arm
        # started from it; arms run on different data start from different draws.
        initials = [block[f"{symbol}_initial"].to_numpy() for block in blocks.values()]
        if initial_guess and all(
            np.allclose(initial, initials[0]) for initial in initials
        ):
            axes.plot(
                rho,
                initials[0] * scale,
                color=initial_color,
                ls=":",
                lw=1.2,
                label="initial guess",
                zorder=1,
            )
        for order, (arm, block) in enumerate(blocks.items()):
            color = arm_colors[order % len(arm_colors)]
            # A Laplace band is centered on its MAP, a sampled band reports its mean.
            if is_laplace[arm]:
                point, label = (
                    "map",
                    ("MAP" if n_laplace == 1 else f"{arm_labels[arm]} MAP"),
                )
            else:
                point, label = "mean", f"{arm_labels[arm]} mean"
            axes.plot(
                block["rho"].to_numpy(),
                block[f"{symbol}_{point}"].to_numpy() * scale,
                color=color,
                lw=1.2,
                label=label,
                zorder=4 if is_laplace[arm] else 3,
            )
            if filled_bands:
                axes.fill_between(
                    block["rho"].to_numpy(),
                    block[f"{symbol}_q05"].to_numpy() * scale,
                    block[f"{symbol}_q95"].to_numpy() * scale,
                    color=color,
                    alpha=0.3,
                    lw=0.0,
                    label=f"{arm_labels[arm]} 5-95%",
                    zorder=2,
                )
            else:
                for quantile, band_label in (
                    ("q05", f"{arm_labels[arm]} 5-95%"),
                    ("q95", None),
                ):
                    axes.plot(
                        block["rho"].to_numpy(),
                        block[f"{symbol}_{quantile}"].to_numpy() * scale,
                        color=color,
                        ls="--",
                        lw=1.0,
                        label=band_label,
                        zorder=2,
                    )
        if points is not None:
            _draw_thomson_points(axes, profile_name, points)
        axes.set_box_aspect(1.0)
        axes.set_xlabel(r"$\rho$")
        axes.set_ylabel(axis_labels.get(symbol, axis_label))
    # On a translucent frame, since the initial guess, a prior draw, can cross any
    # corner and has to stay visible behind the legend.
    axes_array[0].legend(
        loc="lower left",
        ncols=1,
        frameon=True,
        framealpha=0.7,
        edgecolor="none",
        title=_readable_label(title) if title else None,
    )
    return figure


def _draw_prior_band(axes: mpl_axes.Axes, prior: pd.DataFrame, symbol: str) -> None:
    """Draw one panel's prior 5-95% band and median, lighter than any posterior."""
    axes.fill_between(
        prior["rho"].to_numpy(),
        prior[f"{symbol}_q05"].to_numpy(),
        prior[f"{symbol}_q95"].to_numpy(),
        color="0.55",
        alpha=0.18,
        lw=0.0,
        label="prior 5-95% interval",
        zorder=1,
    )
    axes.plot(
        prior["rho"].to_numpy(),
        prior[f"{symbol}_median"].to_numpy(),
        color="0.55",
        ls=":",
        lw=1.1,
        label="prior median",
        zorder=2,
    )


def _readable_label(label: str) -> str:
    """`label` with the underscores outside its math escaped, since Computer Modern text
    has no underscore glyph and sets `_` as a raised dot."""
    return "$".join(
        part if index % 2 else part.replace("_", r"$\_$")
        for index, part in enumerate(label.split("$"))
    )


def _draw_thomson_points(
    axes: mpl_axes.Axes, profile_name: str, points: pd.DataFrame
) -> None:
    """Draw one panel's per-volume Thomson estimates, read off the saved table.

    Goes through `_draw_local_estimate`, so an ablation panel and a recovery figure draw
    the same points the same way.
    """
    _draw_local_estimate(
        axes,
        profile_name,
        points["rho"].to_numpy(),
        thomson_scattering_utils.LocalStateEstimate(
            electron_density=jnp.asarray(points["ne"].to_numpy()),
            electron_density_sigma=jnp.asarray(points["ne_sigma"].to_numpy()),
            electron_temperature_ev=jnp.asarray(points["Te"].to_numpy()),
            electron_temperature_sigma_ev=jnp.asarray(points["Te_sigma"].to_numpy()),
        ),
    )
