# Persists soak-campaign failures to a local results directory:
#
#     <output_root>/<equilibrium_label>/<timestamp>/
#         metadata.json                   # campaign-wide settings
#         failures/shard_<seed0>.parquet  # one per shard with failures
#         no_failures/shard_<seed0>.json  # a clean shard's top-residual points instead
#
# `metadata.json` fixes what is constant across the campaign; each parquet row is one
# failing point, carrying everything needed to reproduce it (`target_xyz` + the
# equilibrium invert it directly; `batch_seed` / `point_index` / `generator` give batch
# context; the `guess_*` seed and the solver's `final_*` iterate describe the miss, and
# `distance_to_boundary` says whether the inside/outside verdict was resolvable at all).

import json
import pathlib

import numpy as np
import pandas as pd

from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.results.inverse_mapping.stress_tests import (
    soak_utils,
)

FAILURE_COLUMNS = (
    "batch_seed",
    "point_index",
    "generator_type",
    "is_outside",
    "mode",
    "target_x",
    "target_y",
    "target_z",
    "guess_rho",
    "guess_theta",
    "final_rho",
    "final_theta",
    "final_residual",
    "distance_to_boundary",
    "failure_kind",
)


def campaign_dir(
    output_root: pathlib.Path, equilibrium_label: str, timestamp: str
) -> pathlib.Path:
    """Directory of a campaign, keyed by the test equilibrium and then its timestamp.

    Args:
        output_root: Root results directory.
        equilibrium_label: Name of the test equilibrium (filesystem-safe).
        timestamp: Launch timestamp, `YYYY-MM-DD-HHMM`.

    Returns:
        The campaign directory path.
    """
    return output_root / equilibrium_label / timestamp


def write_metadata(campaign: pathlib.Path, metadata: dict) -> pathlib.Path:
    """Write the campaign-wide `metadata.json`; return its path.

    Args:
        campaign: The campaign directory.
        metadata: Campaign-wide settings to record.

    Returns:
        The path of the written file.
    """
    campaign.mkdir(parents=True, exist_ok=True)
    path = campaign / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def write_shard_failures(
    campaign: pathlib.Path, shard_id: int, rows: list[dict]
) -> pathlib.Path | None:
    """Write one shard's failing-point rows as parquet; return its path.

    Args:
        campaign: The campaign directory.
        shard_id: The shard's first batch seed, keying its file.
        rows: The failing-point records.

    Returns:
        The parquet path, or None when the shard had no failures (nothing is written).
    """
    if not rows:
        return None
    path = campaign / "failures" / f"shard_{shard_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(FAILURE_COLUMNS)).to_parquet(path, index=False)
    return path


def write_shard_no_failures(
    campaign: pathlib.Path, shard_id: int, rows: list[dict]
) -> pathlib.Path | None:
    """Write a clean shard's highest-residual rows as JSON (same schema as the failures
    parquet); return its path.

    So a shard with zero failures still leaves an inspectable artifact -- its
    worst-converged points -- instead of nothing.

    Args:
        campaign: The campaign directory.
        shard_id: The shard's first batch seed, keying its file.
        rows: The top-residual records.

    Returns:
        The JSON path, or None when there is nothing to write.
    """
    if not rows:
        return None
    path = campaign / "no_failures" / f"shard_{shard_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    return path


def campaign_table(campaign: pathlib.Path) -> pd.DataFrame:
    """Every recorded point of a campaign as one flat table, worst residual first.

    The per-shard artifacts are what a campaign keeps: the failing points of a shard
    that failed, and the worst-converged points of one that did not. This concatenates
    both into a single table, adds the campaign-wide settings from `metadata.json` as
    columns (so a row carries the tolerance it was judged against), and derives the
    cylindrical `r` of each point and a readable `generator` label. It is the table the
    results notebook loads: reading it needs pandas alone, not this package.

    Args:
        campaign: The campaign directory.

    Returns:
        One row per recorded point, sorted by descending `final_residual`.
    """
    metadata = json.loads((campaign / "metadata.json").read_text())
    clean_rows: list[dict] = []
    for path in sorted((campaign / "no_failures").glob("*.json")):
        clean_rows.extend(json.loads(path.read_text()))
    frames = [
        frame
        for frame in (
            read_failures(campaign),
            pd.DataFrame(clean_rows, columns=list(FAILURE_COLUMNS)),
        )
        # An empty frame would make `concat` guess dtypes from nothing.
        if len(frame)
    ]
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "campaign", metadata["timestamp"])
    table.insert(1, "equilibrium", metadata["equilibrium_label"])
    table.insert(2, "n_batches", metadata["n_batches"])
    table.insert(3, "n_points_per_batch", metadata["n_points_per_batch"])
    table.insert(4, "tol", metadata["tol"])
    table.insert(5, "n_shards", metadata.get("n_shards", -1))
    table.insert(
        6, "n_shards_recorded", len(list((campaign / "no_failures").glob("*.json")))
        + len(list((campaign / "failures").glob("*.parquet")))
    )
    table["generator"] = [
        soak_utils.GENERATOR_LABELS[int(value)] for value in table["generator_type"]
    ]
    table["r"] = np.hypot(table["target_x"], table["target_y"])
    return table.sort_values("final_residual", ascending=False, ignore_index=True)


def read_failures(campaign: pathlib.Path) -> pd.DataFrame:
    """Read all shard parquet files of a campaign into one DataFrame.

    Args:
        campaign: The campaign directory.

    Returns:
        The concatenated failing-point rows (empty with the schema columns if none).
    """
    paths = sorted((campaign / "failures").glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=list(FAILURE_COLUMNS))
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def failure_rows(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    test_points: soak_utils.InverseMapTestPoints,
    flux: desc_utils.FluxCoordinates,
    batch_seed: int,
    mode: str,
    tol: float,
) -> list[dict]:
    """Build one record per failing point of a single (batch, mode).

    For each failing point it records the target, the seed the solver started from,
    the solver's final iterate, and a `failure_kind`.

    Args:
        coefficients: The coefficients the inverse map ran on.
        test_points: The generated batch.
        flux: The inverse-map result for the batch.
        batch_seed: The batch's RNG seed (reproduction key).
        mode: `"extrapolated"` or `"confined"`.
        tol: Acceptance tolerance in metres.

    Returns:
        The failing-point records.
    """
    allow_extrapolation = mode == "extrapolated"
    failing, _ = soak_utils.inverse_map_failure_mask(
        coefficients, test_points, flux, tol, allow_extrapolation
    )
    return _rows_for_indices(
        coefficients, test_points, flux, np.flatnonzero(failing), batch_seed, mode, tol
    )


def top_residual_rows(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    test_points: soak_utils.InverseMapTestPoints,
    flux: desc_utils.FluxCoordinates,
    batch_seed: int,
    mode: str,
    count: int = 5,
) -> list[dict]:
    """The `count` points with the highest round-trip residual, in the same schema as
    `failure_rows`.

    Used for a clean shard's `no_failures` file, so a run with zero failures still
    surfaces its worst-converged points. `failure_kind` is `"none"` (these points
    passed).

    Args:
        coefficients: The coefficients the inverse map ran on.
        test_points: The generated batch.
        flux: The inverse-map result for the batch.
        batch_seed: The batch's RNG seed (reproduction key).
        mode: `"extrapolated"` or `"confined"`.
        count: Number of highest-residual points to record.

    Returns:
        The top-residual records.
    """
    residual = np.asarray(flux.residual)
    indices = np.argsort(residual)[::-1][: min(count, residual.size)]
    return _rows_for_indices(
        coefficients, test_points, flux, indices, batch_seed, mode, tol=None
    )


def _rows_for_indices(
    coefficients: desc_types.DescFourierZernikeCoefficients,
    test_points: soak_utils.InverseMapTestPoints,
    flux: desc_utils.FluxCoordinates,
    indices: np.ndarray,
    batch_seed: int,
    mode: str,
    tol: float | None,
) -> list[dict]:
    """Build failure-schema records for the given point indices. With `tol` set each row
    is classified by `_failure_kind`; with `tol=None` the points are not failures and
    `failure_kind` is `"none"`."""
    if indices.size == 0:
        return []
    allow_extrapolation = mode == "extrapolated"
    points = test_points.points_xyz[indices]
    guess_rho, guess_theta = desc_utils.initial_guess(coefficients, points)
    guess_rho = np.asarray(guess_rho)
    guess_theta = np.asarray(guess_theta) % (2.0 * np.pi)
    final_rho = np.asarray(flux.rho)[indices]
    final_theta = np.asarray(flux.theta)[indices]
    final_residual = np.asarray(flux.residual)[indices]
    is_outside = test_points.is_outside[indices]
    # Same quantity the acceptance criterion gates the wrong-side verdict on, recorded per
    # row so a campaign can be re-classified without re-solving.
    distance_to_boundary = soak_utils.boundary_distance(
        coefficients, flux, allow_extrapolation
    )[indices]

    rows = []
    for k, point_index in enumerate(indices):
        rows.append(
            {
                "batch_seed": int(batch_seed),
                "point_index": int(point_index),
                "generator_type": int(test_points.generator_type[point_index]),
                "is_outside": bool(is_outside[k]),
                "mode": mode,
                "target_x": float(points[k, 0]),
                "target_y": float(points[k, 1]),
                "target_z": float(points[k, 2]),
                "guess_rho": float(guess_rho[k]),
                "guess_theta": float(guess_theta[k]),
                "final_rho": float(final_rho[k]),
                "final_theta": float(final_theta[k]),
                "final_residual": float(final_residual[k]),
                "distance_to_boundary": float(distance_to_boundary[k]),
                "failure_kind": (
                    _failure_kind(
                        bool(is_outside[k]),
                        float(final_rho[k]),
                        float(final_residual[k]),
                        float(distance_to_boundary[k]),
                        allow_extrapolation,
                        tol,
                    )
                    if tol is not None
                    else "none"
                ),
            }
        )
    return rows


def _failure_kind(
    is_outside: bool,
    final_rho: float,
    final_residual: float,
    boundary_distance: float,
    allow_extrapolation: bool,
    tol: float,
) -> str:
    """Classify a failure. `rho` is always the true final iterate (never NaN).

    - `inaccurate`: converged above `tol`. Checked first, because a residual above the
      tolerance explains the failure on its own; a non-converged iterate can sit anywhere,
      so reading a side of the boundary off it would say nothing.
    - `wrong_side`: converged, yet on the wrong side of the last closed flux surface and
      far enough from it (`boundary_distance > tol`) for that to be resolvable. The serious
      kind: it satisfies the `residual <= tol` contract, so a caller keying on `rho <= 1`
      is silently given the wrong answer.
    - `at_tolerance`: neither, i.e. the verdict sits inside the map's own accuracy of the
      boundary. Round-off, not a defect; kept as its own label so it is never mistaken for
      `wrong_side`.
    """
    if final_residual >= tol:
        return "inaccurate"
    on_wrong_side = (
        is_outside and final_rho <= 1.0 if allow_extrapolation else is_outside
    )
    if on_wrong_side and boundary_distance > tol:
        return "wrong_side"
    return "at_tolerance"
