"""Interferometer placement by the expected Kullback-Leibler divergence about `ne`, the
best set of parallel chords per poloidal plane, chord direction and chord count.

    python run_study.py
    python run_study.py --chord-counts 1 2 3 4 5 6 --output <CHORD_COUNT_OUTPUT>
"""

import argparse
import functools
import itertools
import json
import math
import pathlib
import sys

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
    interferometer_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_placement_optimization import (  # noqa: E402,E501
    diagnostics_placement_optimization_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E402,E501
    model_storage_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    sweep_utils,
)

TOROIDAL_ANGLES_DEG = tuple(float(angle) for angle in np.linspace(0.0, 60.0, 20))
"""Planes scanned: half a field period, the rest being copies by symmetry."""

DIRECTION_ANGLES_DEG = tuple(float(angle) for angle in np.arange(0.0, 180.0, 15.0))
"""Shared direction of the parallel chords in the `(R, Z)` plane, 0 being horizontal."""

INITIAL_DIRECTION_DEG = 90.0
"""Direction of the unoptimized chords: vertical."""

N_CANDIDATES = 40
"""Parallel chords per plane and direction, spread evenly across the plasma."""

OUTPUT = _MODULE_DIRECTORY / "tables/if_placement_scan.csv"
"""Where the command line writes the scan."""

CHORD_COUNT_OUTPUT = _MODULE_DIRECTORY / "tables/if_placement_scan_chord_counts.csv"
"""Where the scan over one to six chords is written, read by the notebook's last
section."""

LAYOUT_CHUNK = 20_000
"""Layouts scored at once; six of forty candidates are 3.8 million layouts, about 9 GB
of information matrices if scored together."""


def chord_rows(
    z: jt.Float[jax.Array, " n_param"],
    starts: jt.Float[jax.Array, "n_chord 3"],
    ends: jt.Float[jax.Array, "n_chord 3"],
    posterior: posterior_utils.Posterior,
    settings: inference_types.PlasmaEstimationSettings,
    surrogate: model_storage_utils.LoadedSurrogate,
) -> jt.Float[jax.Array, "n_chord n_param"]:
    """Each chord's whitened reading gradient at `z`: its row of the Gauss-Newton
    Jacobian, divided by the constant `settings.interferometer_sigma`.

    Args:
        z: Unconstrained parameters the model is linearized at.
        starts: Start of each chord's in-plasma segment, lab frame, in m.
        ends: End of each chord's in-plasma segment, lab frame, in m.
        posterior: The posterior whose transforms define `z`.
        settings: The inference configuration, for the forward resolutions and the
            chord uncertainty.
        surrogate: The equilibrium forward model.

    Returns:
        One row per chord.
    """
    residuals = functools.partial(
        laplace_approximation_utils.residuals_of_z,
        posterior=posterior,
        settings=settings,
        surrogate=surrogate,
        flux_guess=None,
        fixed_coefficients=None,
    )
    ones = jnp.ones(starts.shape[0])
    readings = residuals(
        z, target=_chord_target(starts, ends, jnp.zeros_like(ones), ones)
    )
    sigmas = interferometer_utils.compute_interferometer_uncertainty(
        readings, settings.interferometer_sigma
    )
    target = _chord_target(starts, ends, readings, sigmas)
    return jax.jacfwd(residuals)(z, target=target)


def scan(
    paths: sweep_utils.SweepPaths,
    toroidal_angles_deg: tuple[float, ...] = TOROIDAL_ANGLES_DEG,
    direction_angles_deg: tuple[float, ...] = DIRECTION_ANGLES_DEG,
    n_candidates: int = N_CANDIDATES,
    truth_names: tuple[str, ...] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
    chord_counts: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    """The best layout of every plane, chord direction and chord count, by the
    information gain about the `ne` parameters averaged over the truths.

    Args:
        paths: Where the dataset and the surrogate live.
        toroidal_angles_deg: Planes to scan.
        direction_angles_deg: Chord directions to scan in every plane.
        n_candidates: Parallel chords per plane and direction.
        truth_names: Ground truths the gain is averaged over.
        chord_counts: Numbers of chords per layout, each searched exhaustively.

    Returns:
        One row per plane, direction and chord count, with the gain per truth, the
        chosen offsets and the chords' surface angles.
    """
    context = _scoring_context(paths, truth_names)
    layouts_of = {
        n_chords: np.array(list(itertools.combinations(range(n_candidates), n_chords)))
        for n_chords in chord_counts
    }
    records = []
    for toroidal_angle_deg in toroidal_angles_deg:
        for direction_angle_deg in direction_angles_deg:
            # The candidate rows are the costly part and every chord count shares them.
            candidates = _candidate_rows(
                context, toroidal_angle_deg, direction_angle_deg, n_candidates
            )
            for n_chords in chord_counts:
                records.append(
                    _best_layout(
                        context,
                        toroidal_angle_deg,
                        direction_angle_deg,
                        candidates,
                        layouts_of[n_chords],
                        truth_names,
                    )
                )
        print(f"[if_placement] {toroidal_angle_deg:.1f} deg done", flush=True)
    return pd.DataFrame(records)


def initial_layout(
    n_chords: int, n_candidates: int = N_CANDIDATES
) -> jt.Int[np.ndarray, " n_chords"]:
    """The unoptimized chords as catalogue indices: one through the middle of the
    plasma, or two at a third and two thirds of its width.

    Args:
        n_chords: Number of chords.
        n_candidates: Parallel chords of the catalogue the chords are taken from.

    Returns:
        The chosen indices.
    """
    return np.round(np.linspace(0, n_candidates - 1, n_chords + 2)[1:-1]).astype(int)


def initial_guess(
    paths: sweep_utils.SweepPaths,
    toroidal_angles_deg: tuple[float, ...],
    n_chords: int,
    n_candidates: int = N_CANDIDATES,
    truth_names: tuple[str, ...] = diagnostics_placement_optimization_utils.TRUTH_NAMES,
) -> pd.DataFrame:
    """The unoptimized layout of `initial_layout`, vertical, scored in each plane.

    Args:
        paths: Where the dataset and the surrogate live.
        toroidal_angles_deg: Planes to score it in.
        n_chords: Number of chords.
        n_candidates: Parallel chords of the catalogue the chords are taken from.
        truth_names: Ground truths the gain is averaged over.

    Returns:
        One row per plane, with the same columns as `scan`.
    """
    context = _scoring_context(paths, truth_names)
    layouts = initial_layout(n_chords, n_candidates)[None, :]
    return pd.DataFrame(
        [
            _best_layout(
                context,
                angle,
                INITIAL_DIRECTION_DEG,
                _candidate_rows(context, angle, INITIAL_DIRECTION_DEG, n_candidates),
                layouts,
                truth_names,
            )
            for angle in toroidal_angles_deg
        ]
    )


def compared_layouts(
    table: pd.DataFrame, paths: sweep_utils.SweepPaths
) -> dict[str, pd.Series]:
    """Per chord count, the optimized layout of the best plane and, of the initial
    guesses in the best and the worst plane, the one with the lowest gain.

    Args:
        table: The scan `scan` wrote.
        paths: Where the dataset and the surrogate live, to score the initial guesses.

    Returns:
        The rows, keyed `<n>_chords_optimized` and `<n>_chords_initial_guess`.
    """
    best = table.loc[
        table.groupby(["toroidal_angle_deg", "n_chords"]).information_gain_bits.idxmax()
    ]
    layouts = {}
    for n_chords, rows in best.groupby("n_chords"):
        optimized = rows.loc[rows.information_gain_bits.idxmax()]
        worst_plane = rows.loc[rows.information_gain_bits.idxmin()]
        guesses = initial_guess(
            paths,
            (
                float(optimized.toroidal_angle_deg),
                float(worst_plane.toroidal_angle_deg),
            ),
            int(n_chords),
        )
        layouts[f"{n_chords}_chords_optimized"] = optimized
        layouts[f"{n_chords}_chords_initial_guess"] = guesses.loc[
            guesses.information_gain_bits.idxmin()
        ]
    return layouts


def design_point(row: pd.Series) -> diagnostic_design_point.DiagnosticDesignPoint:
    """The instrument one scan row describes: its chords alone, no scattering volume,
    the lenses aimed as in the pinned layout so that the layout still builds."""
    pinned = diagnostic_design_point.DiagnosticDesignPoint()
    return diagnostic_design_point.DiagnosticDesignPoint(
        toroidal_angle_deg=float(row.toroidal_angle_deg),
        volume_span_fractions=[],
        optic_aim_fractions=list(pinned.volume_span_fractions),
        chord_surface_angles=json.loads(row.chord_surface_angles),
    )


def _scoring_context(
    paths: sweep_utils.SweepPaths, truth_names: tuple[str, ...]
) -> tuple:
    """Reference geometry, truth points, prior precisions, `ne` indices, jitted rows."""
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
    posterior, priors = diagnostics_placement_optimization_utils.placement_posterior(
        settings, surrogate
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
    interest = diagnostics_placement_optimization_utils.parameter_indices(
        posterior, ("ne_",)
    )
    rows_of = jax.jit(
        functools.partial(
            chord_rows, posterior=posterior, settings=settings, surrogate=surrogate
        )
    )
    return reference, points, precisions, interest, rows_of


def _candidate_rows(
    context: tuple,
    toroidal_angle_deg: float,
    direction_angle_deg: float,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, jt.Float[jax.Array, "n_truth n_candidate n_param"]]:
    """Offsets, surface angles and per-truth Jacobian rows of the parallel candidate
    chords of one plane and direction."""
    reference, points, _, _, rows_of = context
    offsets, surface_angles, starts, ends = (
        diagnostics_placement_optimization_utils.parallel_chords(
            reference.coefficients,
            math.radians(toroidal_angle_deg),
            math.radians(direction_angle_deg),
            n_candidates,
        )
    )
    rows = jnp.stack([rows_of(point, starts, ends) for point in points])
    return offsets, surface_angles, rows


def _best_layout(
    context: tuple,
    toroidal_angle_deg: float,
    direction_angle_deg: float,
    candidates: tuple[np.ndarray, np.ndarray, jax.Array],
    layouts: jt.Int[np.ndarray, "n_layout n_chords_per_layout"],
    truth_names: tuple[str, ...],
) -> dict:
    """The highest-gain layout among `layouts` of one plane and direction, as a row."""
    _, _, precisions, interest, _ = context
    offsets, surface_angles, rows = candidates
    mean_bits = jnp.concatenate(
        [
            jnp.mean(
                _layout_bits(
                    rows[:, layouts[start : start + LAYOUT_CHUNK]], precisions, interest
                ),
                axis=0,
            )
            for start in range(0, len(layouts), LAYOUT_CHUNK)
        ]
    )
    best = int(jnp.argmax(mean_bits))
    bits = _layout_bits(rows[:, layouts[best : best + 1]], precisions, interest)[:, 0]
    return {
        "toroidal_angle_deg": toroidal_angle_deg,
        "direction_angle_deg": direction_angle_deg,
        "n_chords": layouts.shape[1],
        "information_gain_bits": float(jnp.mean(bits)),
        **{f"bits_{name}": float(value) for name, value in zip(truth_names, bits)},
        "offsets_m": json.dumps(offsets[layouts[best]].tolist()),
        "chord_surface_angles": json.dumps(surface_angles[layouts[best]].tolist()),
    }


def _layout_bits(
    chosen: jt.Float[jax.Array, "n_truth n_layout n_chord n_param"],
    precisions: jt.Float[jax.Array, "n_truth n_param n_param"],
    interest: jt.Int[np.ndarray, " n_interest"],
) -> jt.Float[jax.Array, "n_truth n_layout"]:
    """Gain of each layout per truth, its information the rank-one terms of its chords
    summed."""
    information = jnp.einsum("tlci,tlcj->tlij", chosen, chosen)
    return diagnostics_placement_optimization_utils.information_gain_bits(
        information, precisions[:, None], interest
    )


def _chord_target(
    starts: jt.Float[jax.Array, "n_chord 3"],
    ends: jt.Float[jax.Array, "n_chord 3"],
    readings: jt.Float[jax.Array, " n_chord"],
    sigmas: jt.Float[jax.Array, " n_chord"],
) -> inference_target_types.InferenceTarget:
    """An interferometer-only target on the given in-plasma segments."""
    return inference_target_types.InferenceTarget(
        interferometer=inference_target_types.InterferometerTarget(
            geometries=[
                interferometer_utils.build_geometry_from_endpoints(
                    starts[index], ends[index], f"chord_{index}"
                )
                for index in range(starts.shape[0])
            ],
            line_integrated_density=readings,
            line_integrated_density_sigma=sigmas,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sweep_utils.add_sweep_arguments(parser)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--chord-counts", type=int, nargs="+", default=[1, 2])
    arguments = parser.parse_args()
    table = scan(
        sweep_utils.paths_from_arguments(arguments, _MODULE_DIRECTORY.name),
        chord_counts=tuple(arguments.chord_counts),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.output, index=False)
    print(f"[if_placement] {len(table)} rows written to {arguments.output}")


if __name__ == "__main__":
    main()
