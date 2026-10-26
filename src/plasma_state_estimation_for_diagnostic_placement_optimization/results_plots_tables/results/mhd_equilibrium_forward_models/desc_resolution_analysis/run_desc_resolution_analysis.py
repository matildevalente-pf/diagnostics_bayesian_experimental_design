r"""Force-balance residual against DESC resolution, over a sample of equilibria.

Two curves over the same `(L, M, N)` grid.
The first truncates each reference solution to a resolution and recomputes its residual,
so it measures what the smaller expansion can represent; the truncation follows Merlo et
al. (2021, section 2.2.2), who score it by the geometry error. This work's second curve
re-solves the
truncated equilibrium at that resolution, starting from the truncation, so it measures
the residual a dataset generated at that resolution would actually carry.

The scan runs over a random sample of the training split rather than one equilibrium,
so a point on the curve is a population statistic and carries a spread. It writes one
row per equilibrium and resolution; the notebook aggregates and plots.

Every argument has a default, so the command is usually just the path to this script.

    python src/plasma_state_estimation_for_diagnostic_placement_optimization/results_plots_tables/results/mhd_equilibrium_forward_models/desc_resolution_analysis/run_desc_resolution_analysis.py \
        --n-equilibria 30 \
        --n-workers 8

Cost scales as `n_equilibria` times the sixteen grid points, one DESC solve each, so
the default is 480 solves. Each worker holds one whole solve, so `--n-workers` is
bounded by memory rather than by cores. Export `OMP_NUM_THREADS` to give each more
than the one thread set below.
"""

import os

# Set before numpy and DESC are imported, here and in every spawned worker, which
# re-imports this module: the equilibria are shared out over processes, so a threaded
# BLAS underneath each would only oversubscribe the machine. An exported value wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import csv  # noqa: E402
import logging  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

import numpy as np  # noqa: E402

from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (  # noqa: E402,E501
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E402,E501
    dataset_utils,
    evaluation_utils,
)

if TYPE_CHECKING:
    import desc.equilibrium

logger = logging.getLogger(__name__)

RESULTS_FILE = "mhd_resolution_force_residual.csv"
"""The table the resolution figure is drawn from, written beside this script."""

RESOLUTION_GRID: tuple[tuple[int, int, int], ...] = tuple(
    (radial, poloidal, poloidal)
    for radial in (3, 5, 7, 9)
    for poloidal in (4, 6, 8, 9)
)
"""The `(L, M, N)` triples to scan, poloidal and toroidal moving together.

Sixteen points spanning radial 3 to 9 against poloidal and toroidal 4 to 9. A triple is
not ordered by any single mode number, which is why the figure plots against the
coefficient count instead.
"""


def main() -> None:
    arguments = _parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    metadata = dataset_utils.load_metadata(
        arguments.data_dir / f"{arguments.dataset}_settings.json"
    )
    dataset = dataset_utils.load_dataset(
        arguments.data_dir / f"{arguments.dataset}.parquet", metadata
    )
    split = dataset_utils.split_indices(dataset, metadata)
    generator = np.random.default_rng(arguments.seed)
    chosen = generator.choice(
        np.asarray(split.train), size=arguments.n_equilibria, replace=False
    )
    sample = dataset_utils.slice_dataset(dataset, chosen)
    equilibria = dataset_utils.target_equilibria(sample)
    logger.info(
        "%d of %d training equilibria, seed %d, %d resolutions each",
        len(equilibria),
        len(split.train),
        arguments.seed,
        len(RESOLUTION_GRID),
    )

    started = time.perf_counter()
    results_path = arguments.output_dir / RESULTS_FILE
    n_written = 0
    # Written as each equilibrium lands rather than at the end. This runs unattended
    # for hours, and one worker killed for memory breaks the whole pool, so a table
    # appearing only on a clean finish would throw away every equilibrium already paid
    # for. The loader sorts, so the order the rows arrive in does not matter.
    with results_path.open("w", newline="") as stream:
        writer: csv.DictWriter | None = None
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=arguments.n_workers, max_tasks_per_child=1
        ) as pool:
            futures = {
                pool.submit(
                    analyze_equilibrium,
                    equilibrium,
                    equilibrium_id,
                    metadata.basis.resolution,
                ): equilibrium_id
                for equilibrium, equilibrium_id in zip(
                    equilibria, sample.equilibrium_ids, strict=True
                )
            }
            for done, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                try:
                    equilibrium_rows = future.result()
                except Exception as error:  # noqa: BLE001 - one bad equilibrium must not stop the rest
                    logger.warning(
                        "%d/%d  %s  FAILED %s: %s",
                        done,
                        len(futures),
                        futures[future],
                        type(error).__name__,
                        error,
                    )
                    continue
                if writer is None:
                    writer = csv.DictWriter(
                        stream, fieldnames=list(equilibrium_rows[0])
                    )
                    writer.writeheader()
                writer.writerows(equilibrium_rows)
                stream.flush()
                n_written += len(equilibrium_rows)
                logger.info(
                    "%d/%d  %s  reference %.3e  re-solved %.3e to %.3e",
                    done,
                    len(futures),
                    futures[future],
                    equilibrium_rows[0]["fnorm_reference"],
                    min(row["fnorm_resolved"] for row in equilibrium_rows),
                    max(row["fnorm_resolved"] for row in equilibrium_rows),
                )
    logger.info(
        "%d rows from %d equilibria in %.1f min, written to %s",
        n_written,
        n_written // max(len(RESOLUTION_GRID), 1),
        (time.perf_counter() - started) / 60,
        results_path,
    )


def analyze_equilibrium(
    equilibrium: desc_types.DescEquilibrium,
    equilibrium_id: str,
    resolution: desc_types.DescResolution,
) -> list[dict[str, Any]]:
    """Scan the whole resolution grid for one equilibrium.

    Runs in its own process and keeps all sixteen resolutions of one equilibrium
    together, so the reference is built once rather than once per grid point.

    Args:
        equilibrium: The equilibrium as the dataset stored it.
        equilibrium_id: Its id, carried into the table as provenance.
        resolution: The resolution the dataset was solved at, i.e. the reference.

    Returns:
        One row per resolution.
    """
    reference = desc_utils.desc_equilibrium_to_desc_object(equilibrium, resolution)
    reference_residual = evaluation_utils.force_residual(reference)
    reference_coefficients = n_coefficients(reference)

    rows = []
    for radial, poloidal, toroidal in RESOLUTION_GRID:
        truncated = reference.copy()
        # Spectral resolution only: this leaves the collocation grid at the reference's
        # `(22, 22, 24)`, so the residual of every point on the scan is measured on one
        # grid and the re-solve optimizes against the grid the reference was judged on.
        truncated.change_resolution(L=radial, M=poloidal, N=toroidal)
        fnorm_truncated = evaluation_utils.force_residual(truncated)

        prepare_for_solve(truncated)
        started = time.perf_counter()
        truncated.solve(objective="force", verbose=0)
        solve_seconds = time.perf_counter() - started

        rows.append(
            {
                "equilibrium_id": equilibrium_id,
                "radial_resolution": radial,
                "poloidal_resolution": poloidal,
                "toroidal_resolution": toroidal,
                "n_coefficients": n_coefficients(truncated),
                "fnorm_truncated": fnorm_truncated,
                "fnorm_resolved": evaluation_utils.force_residual(truncated),
                "fnorm_reference": reference_residual,
                "reference_n_coefficients": reference_coefficients,
                "solve_seconds": solve_seconds,
            }
        )
    return rows


def prepare_for_solve(equilibrium: "desc.equilibrium.Equilibrium") -> None:
    """Give a truncated equilibrium the two boundary conditions a solve needs, in place.

    `desc_equilibrium_to_desc_object` builds an equilibrium for `compute`, which needs
    neither a magnetic axis nor a boundary surface. Both were measured to be required
    here. Without the axis DESC defaults to `R0 = 10 m` against a real axis at
    `1.027 m`, and the linear-constraint projection drags the geometry before the first
    step. Without the boundary rebuilt under DESC's own orientation convention the solve
    diverges from a start cost of `2.7e3` to `2.7e10` and returns a residual of exactly
    one; with both, the same case converges to `6.8e-3`.

    Args:
        equilibrium: The truncated equilibrium, modified in place.
    """
    import desc.geometry

    axis = equilibrium.get_axis()
    equilibrium.Ra_n = axis.R_n
    equilibrium.Za_n = axis.Z_n
    boundary = equilibrium.get_surface_at(rho=1.0)
    equilibrium.surface = desc.geometry.FourierRZToroidalSurface(
        R_lmn=boundary.R_lmn,
        Z_lmn=boundary.Z_lmn,
        modes_R=boundary.R_basis.modes[:, 1:],
        modes_Z=boundary.Z_basis.modes[:, 1:],
        NFP=equilibrium.NFP,
        sym=equilibrium.sym,
        check_orientation=True,
    )


def n_coefficients(equilibrium: "desc.equilibrium.Equilibrium") -> int:
    """Count the spectral coefficients of an equilibrium, the figure's radial axis.

    Args:
        equilibrium: A live DESC equilibrium.

    Returns:
        The combined width of its `R`, `Z` and `lambda` expansions.
    """
    return int(
        len(equilibrium.R_lmn) + len(equilibrium.Z_lmn) + len(equilibrium.L_lmn)
    )


def _parse_arguments() -> argparse.Namespace:
    """The scan's command line; every argument has a default."""
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-equilibria",
        type=int,
        default=30,
        help="Training equilibria to sample. The cost is this times sixteen solves.",
    )
    parser.add_argument(
        "--dataset",
        default="equilibrium_a_3k_spline_input",
        help="Stem of the dataset files.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=here.parents[5] / "data",
        help="Directory holding the dataset parquet and its settings JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=here,
        help="Directory the results table is written into; the notebook beside this "
        "script reads it from there.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Concurrent equilibria. Bounded by memory, not by cores.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed of the training-split sample."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
