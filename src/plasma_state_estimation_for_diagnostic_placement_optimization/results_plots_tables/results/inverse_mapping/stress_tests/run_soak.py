r"""Process-parallel soak campaign of the inverse map.

Shards are fully independent, so a ProcessPoolExecutor runs them side by side.

Run from the repository root, against which a relative --output-root resolves. The
defaults are the acceptance
campaign on equilibrium A at the machine scale: 200 batches x 5000 points
= 1e6 points over 20 shards, a few minutes on a loaded 32-core box.

    python src/plasma_state_estimation_for_diagnostic_placement_optimization/results_plots_tables/results/inverse_mapping/stress_tests/run_soak.py

The per-shard artifacts stay in the gitignored `results/` tree; the campaign's recorded
points are also flattened into `../inverse_map_soak_residuals.csv`, which is the table
the results notebook loads and plots.
"""

import argparse
import concurrent.futures
import datetime
import multiprocessing
import os
import pathlib
import time

import numpy as np
import pydantic
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.results.inverse_mapping.stress_tests import (
    soak_storage,
    soak_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (
    plots_inverse_mapping,
)

SEED_OF_SEEDS = 42
"""Seed generating the batch seeds, so any batch of this runner reproduces standalone
from its seed."""

DATA_DIR = pathlib.Path(__file__).parents[4] / "inverse_mapping" / "data"
"""Local equilibrium artifacts."""

RESULTS_ROOT = pathlib.Path(__file__).parent / "results"
"""Default output root of the per-shard artifacts; gitignored."""

SUMMARY_CSV = pathlib.Path(__file__).parents[1] / "inverse_map_soak_residuals.csv"
"""Default path of the campaign's flattened point table, in the results folder of the
inverse-mapping area (tracked, and the table the notebook plots)."""


class SoakShardInput(pydantic.BaseModel):
    """One worker's share of the soak campaign: a slice of the batch seeds."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    seeds: list[int]
    """Batch seeds this shard runs, drawn from the campaign's seed-of-seeds."""

    n_points: int
    """Test points per batch."""

    tol: float
    """Acceptance tolerance of the round trip, in metres."""

    maxiter: int
    """Maximum Newton iterations per inversion."""

    coefficients: desc_types.DescFourierZernikeCoefficients
    """Test equilibrium geometry, loaded once and embedded so every shard uses a bit-
    identical equilibrium."""

    campaign_dir: pathlib.Path
    """Directory the shard writes its failures/no-failures artifact to."""


def run_shard(shard: SoakShardInput) -> dict:
    """Run one shard's batches through the round-trip check in extrapolation mode.

    The beyond-boundary map is a strict superset of the confined one (identical for
    interior points, `rho > 1` for outside points), so the campaign only exercises it.
    Compiles the inverse map once, streams the seeds through the round-trip check, and
    writes the collected failing points (not raised) as one parquet file to the
    campaign's `failures/` folder. If the shard has no failures at all, it instead
    writes its five highest-residual points to a `no_failures/` JSON (same schema), so
    a clean shard still leaves something to inspect.

    Args:
        shard: The shard's seeds, settings and equilibrium.

    Returns:
        A summary dict with the shard's batch/failure counts and wall time.
    """
    coefficients = shard.coefficients
    # Solve harder than the acceptance criterion demands: `xyz_to_flux_coordinates`
    # freezes a point as soon as its residual drops below the tolerance it is given, so
    # passing `shard.tol` would park a tail of points immediately below the very
    # threshold they are then judged against (see `SOLVE_TOLERANCE_FACTOR`).
    invert = soak_utils.make_jitted_inverse_map(
        coefficients,
        tol=shard.tol * soak_utils.SOLVE_TOLERANCE_FACTOR,
        maxiter=shard.maxiter,
        allow_extrapolation=True,
    )

    failing_rows: list[dict] = []
    failing_seeds: set[int] = set()
    no_failure_candidates: list[dict] = []
    start = time.monotonic()
    for seed in shard.seeds:
        test_points = soak_utils.sample_inverse_map_test_points(
            coefficients,
            np.random.default_rng(seed),
            shard.n_points,
            boundary_offset=shard.tol,
        )
        flux = invert(test_points.points_xyz)
        rows = soak_storage.failure_rows(
            coefficients,
            test_points,
            flux,
            batch_seed=seed,
            mode="extrapolated",
            tol=shard.tol,
        )
        if rows:
            failing_rows.extend(rows)
            failing_seeds.add(seed)
        else:
            no_failure_candidates.extend(
                soak_storage.top_residual_rows(
                    coefficients,
                    test_points,
                    flux,
                    batch_seed=seed,
                    mode="extrapolated",
                )
            )

    if failing_rows:
        soak_storage.write_shard_failures(
            shard.campaign_dir, shard.seeds[0], failing_rows
        )
    else:
        top_residual = sorted(
            no_failure_candidates, key=lambda row: row["final_residual"], reverse=True
        )[:5]
        soak_storage.write_shard_no_failures(
            shard.campaign_dir, shard.seeds[0], top_residual
        )

    return {
        "shard_id": shard.seeds[0],
        "n_batches": len(shard.seeds),
        "n_failing_points": len(failing_rows),
        "n_failing_batches": len(failing_seeds),
        "elapsed_seconds": time.monotonic() - start,
    }


def main() -> None:
    """Launch the soak campaign locally, one process pool worker per shard."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--equilibrium",
        default=str(DATA_DIR / "desc_equilibrium_a_machine_scale.npz"),
        help="Path to an .npz with the coefficient arrays; the default is "
        "equilibrium A at the machine scale, the one the results notebook reports.",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=200,
        help="Batches of the campaign; with --n-points-per-batch this is the campaign "
        "size, 1e6 points by default.",
    )
    parser.add_argument("--n-points-per-batch", type=int, default=5000)
    parser.add_argument("--batches-per-shard", type=int, default=10)
    parser.add_argument(
        "--n-processes",
        type=int,
        default=_default_n_processes(),
        help="Worker processes. Each runs JAX at OMP_NUM_THREADS=2, so about half the "
        "cores is the useful ceiling; going higher oversubscribes and slows the "
        "campaign down.",
    )
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--output-root", type=pathlib.Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--summary-csv",
        type=pathlib.Path,
        default=SUMMARY_CSV,
        help="Where to flatten the campaign's recorded points to; the results notebook "
        "reads this path. Pass a scratch path for a throwaway run.",
    )
    args = parser.parse_args()

    # Keep each worker's intra-op threads modest so n_processes JAX workers do not
    # oversubscribe the CPU (inherited by the spawned children).
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    coefficients, label, source = _load_equilibrium(args.equilibrium)
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d-%H%M")
    campaign = soak_storage.campaign_dir(args.output_root, label, timestamp)
    fit_l, fit_m, fit_n = soak_utils.FIT_RESOLUTION
    soak_storage.write_metadata(
        campaign,
        {
            "timestamp": timestamp,
            "seed_of_seeds": SEED_OF_SEEDS,
            "n_batches": args.n_batches,
            "n_points_per_batch": args.n_points_per_batch,
            # Recorded so a campaign table can tell a complete campaign from an
            # interrupted one: every shard leaves exactly one artifact.
            "batches_per_shard": args.batches_per_shard,
            "n_shards": -(-args.n_batches // args.batches_per_shard),
            "tol": args.tol,
            "solve_tol": args.tol * soak_utils.SOLVE_TOLERANCE_FACTOR,
            "maxiter": args.maxiter,
            "test_confined": False,
            "test_extrapolation": True,
            "equilibrium_label": label,
            "equilibrium_source": source,
            "fit_resolution": {"L": fit_l, "M": fit_m, "N": fit_n},
        },
    )
    print(f"campaign directory: {campaign}")

    seed_rng = np.random.default_rng(SEED_OF_SEEDS)
    seeds = [int(seed) for seed in seed_rng.integers(0, 2**63, args.n_batches)]
    shards = [
        SoakShardInput(
            seeds=seeds[start : start + args.batches_per_shard],
            n_points=args.n_points_per_batch,
            tol=args.tol,
            maxiter=args.maxiter,
            coefficients=coefficients,
            campaign_dir=campaign,
        )
        for start in range(0, len(seeds), args.batches_per_shard)
    ]

    total_failing_points = 0
    total_failing_batches = 0
    done = 0
    start = time.monotonic()
    # Spawned (not forked) workers, so each starts a fresh JAX runtime.
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.n_processes, mp_context=context
    ) as pool:
        futures = [pool.submit(run_shard, shard) for shard in shards]
        for future in concurrent.futures.as_completed(futures):
            summary = future.result()
            total_failing_points += summary["n_failing_points"]
            total_failing_batches += summary["n_failing_batches"]
            done += 1
            print(
                f"[{done}/{len(shards)}] shard {summary['shard_id']}: "
                f"{summary['n_failing_points']} failing points in "
                f"{summary['n_failing_batches']}/{summary['n_batches']} batches "
                f"({summary['elapsed_seconds']:.0f}s)"
            )

    n_points_total = args.n_batches * args.n_points_per_batch
    rate = total_failing_points / n_points_total if n_points_total else 0.0
    elapsed = time.monotonic() - start
    print(
        f"\ndone in {elapsed / 60:.1f} min: {total_failing_points} failing points "
        f"over {n_points_total:.3g} ({rate:.2g} per point), "
        f"{total_failing_batches}/{args.n_batches} batches with failures"
    )
    failures = soak_storage.read_failures(campaign)
    if len(failures):
        print("failure kinds:")
        print(failures["failure_kind"].value_counts().to_string())
    table = soak_storage.campaign_table(campaign)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.summary_csv, index=False)
    print(
        f"wrote {len(table)} recorded points to {args.summary_csv} "
        f"(worst residual {table['final_residual'].max():.3e} m)"
    )
    print(
        "REPORT row: | "
        f"{timestamp} | {label} | {args.n_batches} x {args.n_points_per_batch} "
        f"(extrapolation) | {total_failing_points} | {rate:.2g} |"
    )


def _default_n_processes(preferred: int = 15, reserved: int = 2) -> int:
    """`preferred` workers when the machine can spare them, else `cores - reserved`.

    A worker is a JAX process at `OMP_NUM_THREADS = 2`, so `preferred` of them want
    `2 * preferred` threads; `reserved` cores are left for the parent and the OS. On a
    machine with fewer than `preferred + reserved` cores this falls back rather than
    oversubscribing.
    """
    return max(1, min(preferred, (os.cpu_count() or 2) - reserved))


def _load_equilibrium(
    equilibrium: str,
) -> tuple[desc_types.DescFourierZernikeCoefficients, str, str]:
    """Resolve the `--equilibrium` path to coefficients, a label, and a source."""
    path = pathlib.Path(equilibrium)
    coefficients = plots_inverse_mapping.load_equilibrium_npz(path).coefficients
    return coefficients, path.stem, str(path)


if __name__ == "__main__":
    main()
