"""Joint interferometer and Thomson placement in study 1's best planes: the laser, its
two optics and a second chord, by the expected Kullback-Leibler divergence.

    python run_scan.py
    python run_scan.py --laser-through-axis --output tables/<name>.csv
    python run_scan.py --laser-through-axis --thomson-only
    python run_scan.py --laser-through-axis --interferometer-only
    python run_scan.py --laser-through-axis --fixed-equilibrium

`--thomson-only` and `--interferometer-only` score each diagnostic alone on the joint
problem's catalogue; `separately_optimized` and `sequentially_optimized` combine them.
`--fixed-equilibrium` linearizes on the reference equilibrium held fixed instead of
the surrogate equilibrium of each plasma state.
"""

import argparse
import functools
import itertools
import json
import math
import pathlib
import sys
from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pandas as pd

_MODULE_DIRECTORY = pathlib.Path(__file__).resolve().parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parents[3]
sys.path.insert(0, str(_PACKAGE_DIRECTORY.parent))

from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E402,E501
    desc_truth_types,
    inference_target_types,
    inference_types,
    laplace_approximation_utils,
    posterior_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E402,E501
    diagnostic_design_point,
    geometry_utils,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_placement_optimization import (  # noqa: E402,E501
    diagnostics_placement_optimization_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (  # noqa: E402
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E402,E501
    model_storage_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.results.diagnostics_placement_optimization.if_placement_study import (  # noqa: E402,E501
    run_study as interferometer_study,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    sweep_utils,
    synthetic_target_utils,
)

N_PLANES = 3
"""Planes the scan covers: study 1's best, ranked by their best two-chord layout."""

VOLUME_FRACTIONS = geometry_utils.uniform_volume_fractions(10)
"""Scattering volumes evenly spaced along the laser's plasma span, never optimized."""

LASER_DIRECTIONS_DEG = (0.0, 30.0, 60.0, 120.0, 150.0)
"""Laser directions in the `(R, Z)` plane.

A vertical laser leaves every volume on one side of the axis, where one of the two
optics would see nothing.
"""

N_LASER_OFFSETS = 10
"""Parallel lasers per direction, spread from the lowest to the highest offset at which
both optics are admissible."""

N_ADMISSIBILITY_OFFSETS = 41
"""Offsets per direction tested for admissible optics, to locate those two edges."""

INITIAL_GUESS_LASER_OFFSETS = 5
"""Parallel lasers per direction across the whole plasma, of which `initial_guess` takes
the horizontal one nearest the design-surface centre."""

N_OPTIC_ANGLES = 2
"""Surface angles tried per optic, spread inside its admissible window."""

N_AXIS_OPTIC_ANGLES_PER_WINDOW = 5
"""Surface angles per contiguous admissible window of an optic for the lasers through
the axis, spread from edge to edge, or all of them if the window holds fewer."""

LAYOUT_FIELDS = (
    "laser_origin",
    "laser_direction",
    "volume_positions",
    "optic_positions",
    "volume_optic_index",
)
"""Thomson geometry fields a layout changes, passed to the compiled Jacobian."""

OUTPUT = _MODULE_DIRECTORY / "tables/if_and_ts_placement_scan.csv"
"""Where the command line writes the scan."""

THOMSON_ONLY_OUTPUT = _MODULE_DIRECTORY / "tables/thomson_placement_scan.csv"
"""Where `--thomson-only` writes the scan of the Thomson system alone, without chords,
which `separately_optimized` and `sequentially_optimized` read."""

FIXED_EQUILIBRIUM_OUTPUT = (
    _MODULE_DIRECTORY / "tables/if_and_ts_placement_scan_fixed_equilibrium.csv"
)
"""Where `--fixed-equilibrium` writes the scan linearized on the reference equilibrium
held fixed, without the surrogate."""

INTERFEROMETER_ONLY_OUTPUT = (
    _MODULE_DIRECTORY / "tables/interferometer_placement_scan.csv"
)
"""Where `--interferometer-only` writes the scan of the chords alone, one through the
axis in each laser direction and a second one from study 1's catalogue, by the gain on
`ne` without the Thomson system, which `separately_optimized` and
`sequentially_optimized` read."""


class _ScoringContext(NamedTuple):
    """What every scoring step shares, built once per scan by `_scoring_context`."""

    reference: desc_truth_types.DescTruth
    """The equilibrium every layout is drawn on."""

    points: list[jt.Float[jax.Array, " n_param"]]
    """The ground truths as unconstrained parameter vectors, where every layout is
    linearized."""

    precisions: jt.Float[jax.Array, "n_truth n_param n_param"]
    """Prior precision at each point."""

    interest: jt.Int[np.ndarray, " n_interest"]
    """The `ne` and `Te` parameters, the objective of the joint problem."""

    density_interest: jt.Int[np.ndarray, " n_density"]
    """The `ne` parameters alone, study 1's objective, for the chords scored without the
    Thomson system."""

    thomson_rows_of: Callable[..., jax.Array]
    """Compiled `thomson_rows` of a point and a layout."""

    chord_rows_of: Callable[..., jax.Array]
    """Compiled `interferometer_study.chord_rows` of a point and chord ends."""


def optic_choices(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: float,
    laser_surface_angles: tuple[float, float],
    edge_to_edge: bool = False,
) -> list[tuple[float, float]]:
    """Forward and backward optic surface angles that keep every volume inside its
    scattering-angle window, `N_OPTIC_ANGLES` per optic inside it or
    `N_AXIS_OPTIC_ANGLES_PER_WINDOW` from edge to edge of each window, every pairing.

    Args:
        coefficients: Equilibrium the laser and the design surface are placed on.
        toroidal_angle: Plane of the layout, in rad.
        laser_surface_angles: The laser's two surface angles, in rad.
        edge_to_edge: Whether to spread the angles from edge to edge of each window.

    Returns:
        The admissible optic pairs; empty if either optic has none.
    """
    volumes, direction, _ = geometry_utils.thomson_volumes_on_chord(
        coefficients,
        toroidal_angle,
        geometry_utils.DEFAULT_SURFACE_OFFSET,
        laser_surface_angles,
        0.0,
        VOLUME_FRACTIONS,
    )
    _, surface = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, geometry_utils.DEFAULT_SURFACE_OFFSET
    )
    is_hfs = volumes[0] < geometry_utils.magnetic_axis(coefficients, toroidal_angle)[0]
    distance = thomson_scattering_utils.collection_object_distance(
        thomson_scattering_utils.W7X_COLLECTION_FOCAL_LENGTH,
        thomson_scattering_utils.W7X_COLLECTION_MAGNIFICATION,
    )
    picks = []
    for served, limits in (
        (is_hfs, geometry_utils.LOW_ANGLE_RANGE_DEG),
        (~is_hfs, geometry_utils.HIGH_ANGLE_RANGE_DEG),
    ):
        admissible = geometry_utils.admissible_surface_angles(
            volumes[:, served], surface, direction, limits, distance
        )
        if admissible.size == 0:
            return []
        if edge_to_edge:
            # Unique indices: a window narrower than the grid gives each angle once.
            picks.append(
                [
                    float(window[index])
                    for window in _windows(admissible)
                    for index in np.unique(
                        np.round(
                            np.linspace(
                                0, window.size - 1, N_AXIS_OPTIC_ANGLES_PER_WINDOW
                            )
                        ).astype(int)
                    )
                ]
            )
        else:
            spread = np.linspace(0, admissible.size - 1, N_OPTIC_ANGLES + 2)[1:-1]
            picks.append(
                [float(admissible[index]) for index in np.round(spread).astype(int)]
            )
    return list(itertools.product(*picks))


def thomson_rows(
    z: jt.Float[jax.Array, " n_param"],
    layout: dict[str, jt.Shaped[jax.Array, "..."]],
    template: inference_target_types.ThomsonSignalTarget,
    posterior: posterior_utils.Posterior,
    settings: inference_types.PlasmaEstimationSettings,
    surrogate: model_storage_utils.LoadedSurrogate,
    signal_scale: float,
    noise_model: thomson_scattering_utils.ThomsonNoiseModel,
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> jt.Float[jax.Array, "n_signal n_param"]:
    """Whitened Jacobian of one layout's Thomson channels at `z`, their uncertainties
    evaluated on the signals predicted at `z`; `template` fixes the shapes, so the
    layout enters as arrays and the function compiles once.

    Args:
        z: Unconstrained parameters the model is linearized at.
        layout: The `LAYOUT_FIELDS` of the layout's geometry, its band scales and its
            nominal calibration.
        template: A Thomson target with the scan's volume and channel counts.
        posterior: The posterior whose transforms define `z`.
        settings: The inference configuration, for the forward resolutions.
        surrogate: The equilibrium forward model.
        signal_scale: `kappa` in V s per photoelectron.
        noise_model: Uncertainty model of the Thomson channels.
        fixed_coefficients: Geometry to hold frozen; None predicts it per `z`.

    Returns:
        One row per channel of every volume.
    """
    signals = template.model_copy(
        update={
            "geometry": template.geometry.model_copy(
                update={name: layout[name] for name in LAYOUT_FIELDS}
            ),
            "response": template.response.model_copy(
                update={"band_scales": layout["band_scales"]}
            ),
            "global_calibration": layout["global_calibration"],
        }
    )
    residuals = functools.partial(
        laplace_approximation_utils.residuals_of_z,
        posterior=posterior,
        settings=settings,
        surrogate=surrogate,
        flux_guess=None,
        fixed_coefficients=fixed_coefficients,
    )
    # With zero signals and unit uncertainties the residuals are the predictions.
    predicted = residuals(
        z, target=inference_target_types.InferenceTarget(thomson_signals=signals)
    )
    uncertainties = thomson_scattering_utils.signal_uncertainties(
        predicted.reshape(template.signals.shape), noise_model, signal_scale
    )
    target = inference_target_types.InferenceTarget(
        thomson_signals=signals.model_copy(update={"uncertainties": uncertainties})
    )
    return jax.jacfwd(residuals)(z, target=target)


def best_planes(n_planes: int = N_PLANES) -> tuple[float, ...]:
    """Study 1's `n_planes` best planes, by the gain of their best two-chord layout.

    Args:
        n_planes: Number of planes to return.

    Returns:
        Their toroidal angles in deg, best first.
    """
    table = pd.read_csv(interferometer_study.OUTPUT)
    pairs = table[table.n_chords == 2]
    best = pairs.groupby("toroidal_angle_deg").information_gain_bits.max()
    return tuple(float(angle) for angle in best.nlargest(n_planes).index)


def scan(
    paths: sweep_utils.SweepPaths,
    toroidal_angles_deg: tuple[float, ...],
    truth_names: tuple[str, ...] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
    laser_through_axis: bool = False,
    thomson_only: bool = False,
    interferometer_only: bool = False,
    fixed_equilibrium: bool = False,
) -> pd.DataFrame:
    """Every laser and admissible optic pair in each plane, with one chord on the laser
    and the second chord that raises the gain most.

    Args:
        paths: Where the dataset and the surrogate live.
        toroidal_angles_deg: Planes to scan.
        truth_names: Ground truths the gain is averaged over.
        laser_through_axis: Whether each laser direction has one laser, through the
            reference magnetic axis, instead of `N_LASER_OFFSETS` admissible ones,
            with its optics spread from edge to edge of their windows.
        thomson_only: Whether to score the Thomson system alone, without chords.
        interferometer_only: Whether to score the chords alone, one through the axis
            in each laser direction and the second from the catalogue, by the gain on
            `ne` without the Thomson system.
        fixed_equilibrium: Whether to linearize on the reference equilibrium held
            fixed instead of the surrogate equilibrium of each plasma state.

    Returns:
        One row per plane, laser and optic pair, or per plane and axis chord with
        `interferometer_only`: the gains and the layout.
    """
    context = _scoring_context(paths, truth_names, fixed_equilibrium)
    records = []
    for toroidal_angle_deg in toroidal_angles_deg:
        chords, lasers = _catalogues(context, toroidal_angle_deg, laser_through_axis)
        if interferometer_only:
            records.extend(
                _chord_only_records(
                    context, toroidal_angle_deg, chords, lasers, truth_names
                )
            )
            print(
                f"[if_and_ts_placement] {toroidal_angle_deg:.1f} deg done", flush=True
            )
            continue
        for laser in range(lasers[0].size):
            coincident = _coincident(chords, lasers, laser)
            for optic_pair in optic_choices(
                context.reference.coefficients,
                math.radians(toroidal_angle_deg),
                tuple(lasers[2][laser].tolist()),
                edge_to_edge=laser_through_axis,
            ):
                try:
                    thomson = _thomson_information(
                        context,
                        _laser_design(
                            toroidal_angle_deg, lasers[2][laser].tolist(), optic_pair
                        ),
                    )
                except ValueError as error:
                    print(f"[if_and_ts_placement] skipped: {error}", flush=True)
                    continue
                if thomson_only:
                    bits = (
                        diagnostics_placement_optimization_utils.information_gain_bits(
                            thomson, context.precisions, context.interest
                        )
                    )
                    records.append(
                        _record(
                            toroidal_angle_deg,
                            None,
                            None,
                            lasers,
                            laser,
                            optic_pair,
                            np.asarray(bits),
                            truth_names,
                        )
                    )
                    continue
                bits = _chord_bits(context, thomson, chords, lasers, laser)
                mean = np.where(coincident, -np.inf, np.mean(bits, axis=0))
                chord = int(np.argmax(mean))
                records.append(
                    _record(
                        toroidal_angle_deg,
                        chords,
                        chord,
                        lasers,
                        laser,
                        optic_pair,
                        bits[:, chord],
                        truth_names,
                    )
                )
        print(f"[if_and_ts_placement] {toroidal_angle_deg:.1f} deg done", flush=True)
    return pd.DataFrame(records)


def initial_guess(
    paths: sweep_utils.SweepPaths,
    truth_names: tuple[str, ...] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
) -> pd.Series:
    """Nothing optimized, scored like a `scan` row: in the first plane, the horizontal
    laser nearest the plasma centre with its first admissible optic pair, the chord on
    the laser and study 1's vertical chord through the middle of the plasma.

    Args:
        paths: Where the dataset and the surrogate live.
        truth_names: Ground truths the gain is averaged over.

    Returns:
        The layout and its gains, with the columns of `scan`.
    """
    context = _scoring_context(paths, truth_names)
    toroidal_angle_deg = interferometer_study.TOROIDAL_ANGLES_DEG[0]
    chords, lasers = _catalogues(context, toroidal_angle_deg, initial_guess=True)
    laser = int(np.argmin(np.where(lasers[0] == 0.0, np.abs(lasers[1]), np.inf)))
    chord = int(
        np.flatnonzero(chords[0] == interferometer_study.INITIAL_DIRECTION_DEG)[
            interferometer_study.initial_layout(1)[0]
        ]
    )
    optic_pair = optic_choices(
        context.reference.coefficients,
        math.radians(toroidal_angle_deg),
        tuple(lasers[2][laser].tolist()),
    )[0]
    thomson = _thomson_information(
        context,
        _laser_design(toroidal_angle_deg, lasers[2][laser].tolist(), optic_pair),
    )
    bits = _chord_bits(context, thomson, chords, lasers, laser)
    return pd.Series(
        _record(
            toroidal_angle_deg,
            chords,
            chord,
            lasers,
            laser,
            optic_pair,
            bits[:, chord],
            truth_names,
        )
    )


def separately_optimized(
    paths: sweep_utils.SweepPaths,
    truth_names: tuple[str, ...] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
    context: _ScoringContext | None = None,
) -> pd.Series:
    """The two diagnostics optimized apart on the joint problem's catalogue and then
    combined, each in its own plane: the best Thomson system of the `--thomson-only`
    scan and the best chords of the `--interferometer-only` scan, scored together like a
    `scan` row.

    Args:
        paths: Where the dataset and the surrogate live.
        truth_names: Ground truths the gain is averaged over.
        context: A scoring context of `truth_names` to reuse; None builds one.

    Returns:
        The layout and its gains, with the columns of `scan`, the chords' plane and
        the direction of their axis chord.
    """
    chords = pd.read_csv(INTERFEROMETER_ONLY_OUTPUT)
    best = chords.loc[chords.information_gain_bits.idxmax()]
    return _combined(paths, _best_thomson_system(), best, True, truth_names, context)


def sequentially_optimized(
    paths: sweep_utils.SweepPaths,
    truth_names: tuple[str, ...] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
    context: _ScoringContext | None = None,
) -> pd.Series:
    """The Thomson system optimized alone and the chords placed after it, in its plane
    with the first one along its laser, by the objective of the `--interferometer-only`
    scan: that scan's row on the best Thomson system's plane and laser, scored
    together like a `scan` row.

    Args:
        paths: Where the dataset and the surrogate live.
        truth_names: Ground truths the gain is averaged over.
        context: A scoring context of `truth_names` to reuse; None builds one.

    Returns:
        The layout and its gains, with the columns of `scan` and the direction of the
        axis chord.
    """
    thomson = _best_thomson_system()
    chords = pd.read_csv(INTERFEROMETER_ONLY_OUTPUT)
    match = chords[
        np.isclose(chords.toroidal_angle_deg, thomson.toroidal_angle_deg)
        & np.isclose(chords.laser_direction_deg, thomson.laser_direction_deg)
    ]
    if len(match) != 1:
        raise ValueError(
            "the interferometer-only scan holds no axis chord in the best Thomson "
            f"system's plane and direction, {thomson.toroidal_angle_deg:.1f} deg at "
            f"{thomson.laser_direction_deg:.0f} deg; rerun it on the same planes"
        )
    axis_chord = json.loads(match.iloc[0].laser_surface_angles)
    laser = json.loads(thomson.laser_surface_angles)
    if not np.allclose(axis_chord, laser, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"the axis chord {axis_chord} does not lie on the laser {laser}; the two "
            "scans were placed on different reference equilibria"
        )
    return _combined(paths, thomson, match.iloc[0], False, truth_names, context)


def layout_gains(
    paths: sweep_utils.SweepPaths,
    layouts: Mapping[str, pd.Series],
    truth_names: tuple[str, ...] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
    context: _ScoringContext | None = None,
) -> pd.DataFrame:
    """Gain of each layout read off its design point, for layouts no scan produced or
    for truths no scan averaged over.

    Args:
        paths: Where the dataset and the surrogate live.
        layouts: Rows with the columns `design_point` reads, by name.
        truth_names: Ground truths the gain is averaged over.
        context: A scoring context of `truth_names` to reuse; None builds one.

    Returns:
        One row per layout: `information_gain_bits` and `bits_<truth>`.
    """
    if context is None:
        context = _scoring_context(paths, truth_names)
    gains = {}
    for name, row in layouts.items():
        bits = _design_bits(context, design_point(row))
        gains[name] = {
            "information_gain_bits": float(np.mean(bits)),
            **{
                f"bits_{truth}": float(value) for truth, value in zip(truth_names, bits)
            },
        }
    return pd.DataFrame.from_dict(gains, orient="index")


def compared_layouts(
    table: pd.DataFrame, paths: sweep_utils.SweepPaths
) -> dict[str, pd.Series]:
    """The unoptimized layout of `initial_guess`, the scan's best row,
    `separately_optimized` and `sequentially_optimized`.

    Args:
        table: The scan `scan` wrote.
        paths: Where the dataset and the surrogate live, to score the other layouts.

    Returns:
        The rows by layout name.
    """
    optimized = table.loc[table.information_gain_bits.idxmax()]
    context = _scoring_context(
        paths, diagnostics_placement_optimization_utils.TRUTH_NAMES
    )
    return {
        "initial_guess": initial_guess(paths),
        "optimized": optimized,
        "separately_optimized": separately_optimized(paths, context=context),
        "sequentially_optimized": sequentially_optimized(paths, context=context),
    }


def design_point(row: pd.Series) -> diagnostic_design_point.DiagnosticDesignPoint:
    """The instrument one scan row describes."""
    interferometer_plane = row.get("interferometer_toroidal_angle_deg")
    return diagnostic_design_point.DiagnosticDesignPoint(
        toroidal_angle_deg=float(row.toroidal_angle_deg),
        interferometer_toroidal_angle_deg=(
            None
            if interferometer_plane is None or pd.isna(interferometer_plane)
            else float(interferometer_plane)
        ),
        laser_surface_angles=json.loads(row.laser_surface_angles),
        optic_surface_angles=json.loads(row.optic_surface_angles),
        volume_span_fractions=json.loads(row.volume_span_fractions),
        chord_surface_angles=json.loads(row.chord_surface_angles),
    )


def _scoring_context(
    paths: sweep_utils.SweepPaths,
    truth_names: tuple[str, ...],
    fixed_equilibrium: bool = False,
) -> _ScoringContext:
    """Reference, truth points, prior precisions, interest indices and the compiled
    Thomson and chord Jacobians, on one posterior with a calibration per volume; with
    `fixed_equilibrium`, both Jacobians are taken on the reference equilibrium."""
    parquet_path = paths.data_directory / f"{paths.dataset}.parquet"
    settings_path = paths.data_directory / f"{paths.dataset}_settings.json"
    reference = sweep_utils.reference_truth(parquet_path, settings_path)
    surrogate = model_storage_utils.load_surrogate(
        paths.models_directory,
        paths.model_name,
        paths.data_directory,
        machine_scale=reference.machine_scale,
    )
    settings = inference_types.PlasmaEstimationSettings()
    design = diagnostic_design_point.DiagnosticDesignPoint(
        volume_span_fractions=list(VOLUME_FRACTIONS)
    )
    placement = synthetic_target_utils.build_diagnostic_placement(
        design, reference, verbose=False
    )
    geometry = placement.thomson_geometry
    assert geometry is not None, "the design point carries volumes"
    shape = (
        len(VOLUME_FRACTIONS),
        len(placement.polychromator_response.filter_centers),
    )
    template = inference_target_types.ThomsonSignalTarget(
        geometry=geometry,
        response=placement.polychromator_response,
        signals=jnp.zeros(shape),
        uncertainties=jnp.ones(shape),
        global_calibration=thomson_scattering_utils.absolute_calibration(geometry),
    )
    posterior, priors = diagnostics_placement_optimization_utils.placement_posterior(
        settings, surrogate, template
    )
    points = [
        diagnostics_placement_optimization_utils.truth_point(
            desc_truth_types.read_desc_truth(name, parquet_path, settings_path),
            posterior,
        )
        for name in truth_names
    ]
    precisions = jnp.stack(
        [
            laplace_approximation_utils.prior_precision(point, posterior, priors)
            for point in points
        ]
    )
    interest = np.array(
        [
            index
            for index, site in enumerate(posterior.site_names)
            if site.startswith(("ne_", "te_"))
        ]
    )
    # The reference equilibrium is also the fixed arm of the self-consistency study, so
    # the fixed scan and the fixed inference share one geometry.
    fixed_coefficients = reference.coefficients if fixed_equilibrium else None
    thomson_rows_of = jax.jit(
        functools.partial(
            thomson_rows,
            template=template,
            posterior=posterior,
            settings=settings,
            surrogate=surrogate,
            signal_scale=placement.thomson_signal_scale,
            noise_model=design.thomson_noise_model,
            fixed_coefficients=fixed_coefficients,
        )
    )
    chord_rows_of = jax.jit(
        functools.partial(
            interferometer_study.chord_rows,
            posterior=posterior,
            settings=settings,
            surrogate=surrogate,
            fixed_coefficients=fixed_coefficients,
        )
    )
    return _ScoringContext(
        reference=reference,
        points=points,
        precisions=precisions,
        interest=interest,
        density_interest=np.array(
            [
                index
                for index, site in enumerate(posterior.site_names)
                if site.startswith(("ne_",))
            ]
        ),
        thomson_rows_of=thomson_rows_of,
        chord_rows_of=chord_rows_of,
    )


def _catalogues(
    context: _ScoringContext,
    toroidal_angle_deg: float,
    laser_through_axis: bool = False,
    initial_guess: bool = False,
) -> tuple[tuple, tuple]:
    """Study 1's chord catalogue and the laser catalogue of one plane; `initial_guess`
    spreads the lasers across the whole plasma instead of their admissible range."""
    return (
        _chord_catalogue(
            context,
            toroidal_angle_deg,
            interferometer_study.DIRECTION_ANGLES_DEG,
            interferometer_study.N_CANDIDATES,
        ),
        _chord_catalogue(
            context,
            toroidal_angle_deg,
            LASER_DIRECTIONS_DEG,
            INITIAL_GUESS_LASER_OFFSETS if initial_guess else N_LASER_OFFSETS,
            through_axis=laser_through_axis,
            admissible=not initial_guess,
        ),
    )


def _chord_catalogue(
    context: _ScoringContext,
    toroidal_angle_deg: float,
    directions_deg: tuple[float, ...],
    n_chords: int,
    through_axis: bool = False,
    admissible: bool = False,
) -> tuple[
    jt.Float[np.ndarray, " n_chord"],
    jt.Float[np.ndarray, " n_chord"],
    jt.Float[np.ndarray, "n_chord 2"],
    jt.Float[jax.Array, "n_truth n_chord n_param"],
]:
    """`n_chords` parallel chords per direction, across the plasma or across the offsets
    with admissible optics, or the one through the magnetic axis: direction, offset,
    surface angles, and each chord's whitened Jacobian row at every truth."""
    coefficients = context.reference.coefficients
    toroidal_angle = math.radians(toroidal_angle_deg)
    directions, offsets, surface_angles, rows = [], [], [], []
    for direction_deg in directions_deg:
        direction = math.radians(direction_deg)
        if through_axis:
            axis_offset = _axis_offset(coefficients, toroidal_angle, direction)
            offset, angles, starts, ends = (
                diagnostics_placement_optimization_utils.chords_at_offsets(
                    coefficients, toroidal_angle, direction, np.array([axis_offset])
                )
            )
        elif admissible:
            offset, angles, starts, ends = (
                diagnostics_placement_optimization_utils.chords_at_offsets(
                    coefficients,
                    toroidal_angle,
                    direction,
                    _admissible_offsets(
                        coefficients, toroidal_angle, direction, n_chords
                    ),
                )
            )
        else:
            offset, angles, starts, ends = (
                diagnostics_placement_optimization_utils.parallel_chords(
                    coefficients, toroidal_angle, direction, n_chords
                )
            )
        directions.append(np.full(offset.size, direction_deg))
        offsets.append(offset)
        surface_angles.append(angles)
        rows.append(
            jnp.stack(
                [context.chord_rows_of(point, starts, ends) for point in context.points]
            )
        )
    return (
        np.concatenate(directions),
        np.concatenate(offsets),
        np.concatenate(surface_angles),
        jnp.concatenate(rows, axis=1),
    )


def _admissible_offsets(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: float,
    direction_angle: float,
    n_lasers: int,
) -> jt.Float[np.ndarray, " n_laser"]:
    """`n_lasers` offsets spread from the lowest to the highest of
    `N_ADMISSIBILITY_OFFSETS` test offsets at which both optics are admissible."""
    offsets, surface_angles, _, _ = (
        diagnostics_placement_optimization_utils.parallel_chords(
            coefficients, toroidal_angle, direction_angle, N_ADMISSIBILITY_OFFSETS
        )
    )
    admitted = offsets[
        [
            bool(optic_choices(coefficients, toroidal_angle, tuple(angles.tolist())))
            for angles in surface_angles
        ]
    ]
    return np.linspace(admitted.min(), admitted.max(), n_lasers)


def _windows(
    angles: jt.Float[np.ndarray, " n_angle"],
) -> list[jt.Float[np.ndarray, " n_window_angle"]]:
    """Contiguous runs of sorted grid angles, the run across +-pi joined into one."""
    if angles.size < 2:
        return [angles]
    gap = 1.5 * np.diff(angles).min()
    windows = np.split(angles, np.flatnonzero(np.diff(angles) > gap) + 1)
    if len(windows) > 1 and angles[0] + 2 * math.pi - angles[-1] <= gap:
        windows = [np.concatenate([windows[-1], windows[0]]), *windows[1:-1]]
    return windows


def _axis_offset(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    toroidal_angle: float,
    direction_angle: float,
) -> float:
    """Offset from the design-surface centre of the line along `direction_angle` through
    the magnetic axis, in the convention of `chords_at_offsets`."""
    _, surface = geometry_utils.cross_section_and_surface(
        coefficients, toroidal_angle, geometry_utils.DEFAULT_SURFACE_OFFSET
    )
    normal = np.array([-math.sin(direction_angle), math.cos(direction_angle)])
    axis = geometry_utils.magnetic_axis(coefficients, toroidal_angle)
    return float(normal @ (axis - np.asarray(surface.centre)))


def _laser_design(
    toroidal_angle_deg: float,
    laser_surface_angles: list[float],
    optic_pair: tuple[float, float],
) -> diagnostic_design_point.DiagnosticDesignPoint:
    """The scan's Thomson system on one laser, with the chord along the laser."""
    return diagnostic_design_point.DiagnosticDesignPoint(
        toroidal_angle_deg=toroidal_angle_deg,
        laser_surface_angles=laser_surface_angles,
        optic_surface_angles=list(optic_pair),
        volume_span_fractions=list(VOLUME_FRACTIONS),
        chord_surface_angles=[laser_surface_angles],
    )


def _thomson_information(
    context: _ScoringContext, design: diagnostic_design_point.DiagnosticDesignPoint
) -> jt.Float[jax.Array, "n_truth n_param n_param"]:
    """Gauss-Newton information `J^T J` of a design's Thomson channels per truth."""
    placement = synthetic_target_utils.build_diagnostic_placement(
        design, context.reference, verbose=False
    )
    geometry = placement.thomson_geometry
    assert geometry is not None, "the design point carries volumes"
    layout = {name: jnp.asarray(getattr(geometry, name)) for name in LAYOUT_FIELDS}
    layout["band_scales"] = jnp.asarray(placement.polychromator_response.band_scales)
    layout["global_calibration"] = thomson_scattering_utils.absolute_calibration(
        geometry
    )
    thomson = [context.thomson_rows_of(point, layout) for point in context.points]
    return jnp.stack([rows.T @ rows for rows in thomson])


def _chord_bits(
    context: _ScoringContext,
    thomson: jt.Float[jax.Array, "n_truth n_param n_param"],
    chords: tuple,
    lasers: tuple,
    laser: int,
    interest: jt.Int[np.ndarray, " n_interest"] | None = None,
) -> jt.Float[np.ndarray, "n_truth n_chord"]:
    """Gain per truth of the Thomson information `thomson`, the chord along `laser` and
    each catalogue chord as the second one, on `context.interest` unless `interest` says
    otherwise."""
    base = thomson + jnp.einsum("ti,tj->tij", lasers[3][:, laser], lasers[3][:, laser])
    information = base[:, None] + jnp.einsum("tci,tcj->tcij", chords[3], chords[3])
    return np.asarray(
        diagnostics_placement_optimization_utils.information_gain_bits(
            information,
            context.precisions[:, None],
            context.interest if interest is None else interest,
        )
    )


def _coincident(
    chords: tuple, lasers: tuple, laser: int
) -> jt.Bool[np.ndarray, " n_chord"]:
    """Which catalogue chords run along `laser`, where the first chord already lies, so
    that none of them may serve as the second one."""
    return (chords[0] == lasers[0][laser]) & np.isclose(chords[1], lasers[1][laser])


def _chord_only_records(
    context: _ScoringContext,
    toroidal_angle_deg: float,
    chords: tuple,
    lasers: tuple,
    truth_names: tuple[str, ...],
) -> list[dict]:
    """One row per axis chord of `lasers`: the catalogue chord that, as the second one,
    raises the gain on `ne` most, with no Thomson system in the objective."""
    no_thomson = jnp.zeros_like(context.precisions)
    records = []
    for laser in range(lasers[0].size):
        bits = _chord_bits(
            context, no_thomson, chords, lasers, laser, context.density_interest
        )
        mean = np.where(
            _coincident(chords, lasers, laser), -np.inf, np.mean(bits, axis=0)
        )
        chord = int(np.argmax(mean))
        records.append(
            _record(
                toroidal_angle_deg,
                chords,
                chord,
                lasers,
                laser,
                None,
                bits[:, chord],
                truth_names,
            )
        )
    return records


def _best_thomson_system() -> pd.Series:
    """The `--thomson-only` scan's best row, the Thomson system optimized alone."""
    thomson = pd.read_csv(THOMSON_ONLY_OUTPUT)
    return thomson.loc[thomson.information_gain_bits.idxmax()].copy()


def _combined(
    paths: sweep_utils.SweepPaths,
    thomson: pd.Series,
    chords: pd.Series,
    own_plane: bool,
    truth_names: tuple[str, ...],
    context: _ScoringContext | None,
) -> pd.Series:
    """The Thomson system of `thomson` with the chords of `chords`, in the chords' own
    plane or in the Thomson plane, scored with both diagnostics like a `scan` row."""
    row = thomson.copy()
    row["interferometer_toroidal_angle_deg"] = (
        float(chords.toroidal_angle_deg) if own_plane else math.nan
    )
    row["axis_chord_direction_deg"] = float(chords.laser_direction_deg)
    row["chord_direction_deg"] = float(chords.chord_direction_deg)
    row["chord_offset_m"] = float(chords.chord_offset_m)
    row["chord_surface_angles"] = chords.chord_surface_angles
    row.update(
        layout_gains(paths, {"combined": row}, truth_names, context=context).iloc[0]
    )
    return row


def _design_bits(
    context: _ScoringContext, design: diagnostic_design_point.DiagnosticDesignPoint
) -> jt.Float[np.ndarray, " n_truth"]:
    """Gain per truth of a design point's Thomson channels and chords together, the
    chords trimmed to the plasma as the inference trims them."""
    coefficients = context.reference.coefficients
    information = (
        _thomson_information(context, design)
        if design.volume_span_fractions
        else jnp.zeros_like(context.precisions)
    )
    _, _, chords = diagnostic_design_point.build_diagnostic_geometry(
        coefficients, design
    )
    if chords:
        trimmed = geometry_utils.trim_chords_to_plasma(chords, coefficients)
        starts = jnp.stack([jnp.asarray(chord.chord_start) for chord in trimmed])
        ends = jnp.stack([jnp.asarray(chord.chord_end) for chord in trimmed])
        rows = jnp.stack(
            [context.chord_rows_of(point, starts, ends) for point in context.points]
        )
        information = information + jnp.einsum("tci,tcj->tij", rows, rows)
    return np.asarray(
        diagnostics_placement_optimization_utils.information_gain_bits(
            information, context.precisions, context.interest
        )
    )


def _record(
    toroidal_angle_deg: float,
    chords: tuple | None,
    chord: int | None,
    lasers: tuple,
    laser: int,
    optic_pair: tuple[float, float] | None,
    bits: jt.Float[np.ndarray, " n_truth"],
    truth_names: tuple[str, ...],
) -> dict:
    """One scan row: the gains and the layout that reached them; `chord` None for the
    Thomson system alone, `optic_pair` None for the chords alone, whose laser columns
    then describe the axis chord."""
    if chords is None or chord is None:
        second = {
            "chord_direction_deg": math.nan,
            "chord_offset_m": math.nan,
            "chord_surface_angles": json.dumps([]),
        }
    else:
        second = {
            "chord_direction_deg": float(chords[0][chord]),
            "chord_offset_m": float(chords[1][chord]),
            "chord_surface_angles": json.dumps(
                [lasers[2][laser].tolist(), chords[2][chord].tolist()]
            ),
        }
    return {
        "toroidal_angle_deg": toroidal_angle_deg,
        "laser_direction_deg": float(lasers[0][laser]),
        "laser_offset_m": float(lasers[1][laser]),
        **second,
        "information_gain_bits": float(np.mean(bits)),
        **{f"bits_{name}": float(value) for name, value in zip(truth_names, bits)},
        "laser_surface_angles": json.dumps(lasers[2][laser].tolist()),
        "optic_surface_angles": json.dumps(
            [] if optic_pair is None else list(optic_pair)
        ),
        "volume_span_fractions": json.dumps(
            [] if optic_pair is None else list(VOLUME_FRACTIONS)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sweep_utils.add_sweep_arguments(parser)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="defaults to THOMSON_ONLY_OUTPUT, INTERFEROMETER_ONLY_OUTPUT, "
        "FIXED_EQUILIBRIUM_OUTPUT or OUTPUT",
    )
    parser.add_argument(
        "--laser-through-axis",
        action="store_true",
        help="one laser per direction, through the reference magnetic axis, with "
        "its optic angles spread from edge to edge of each admissible window",
    )
    parser.add_argument(
        "--thomson-only",
        action="store_true",
        help="score the Thomson system alone, without chords",
    )
    parser.add_argument(
        "--interferometer-only",
        action="store_true",
        help="score the chords alone, one through the axis in each laser direction "
        "and a second from the catalogue, by the gain on ne without the Thomson system",
    )
    parser.add_argument(
        "--fixed-equilibrium",
        action="store_true",
        help="linearize on the reference equilibrium held fixed, without the surrogate",
    )
    arguments = parser.parse_args()
    output = arguments.output or (
        THOMSON_ONLY_OUTPUT
        if arguments.thomson_only
        else INTERFEROMETER_ONLY_OUTPUT
        if arguments.interferometer_only
        else FIXED_EQUILIBRIUM_OUTPUT
        if arguments.fixed_equilibrium
        else OUTPUT
    )
    table = scan(
        sweep_utils.paths_from_arguments(arguments, _MODULE_DIRECTORY.name),
        best_planes(),
        laser_through_axis=arguments.laser_through_axis,
        thomson_only=arguments.thomson_only,
        interferometer_only=arguments.interferometer_only,
        fixed_equilibrium=arguments.fixed_equilibrium,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"[if_and_ts_placement] {len(table)} rows written to {output}")


if __name__ == "__main__":
    main()
