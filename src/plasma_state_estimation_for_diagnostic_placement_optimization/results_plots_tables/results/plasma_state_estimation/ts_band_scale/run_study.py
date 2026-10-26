r"""Does scaling the polychromator passbands to the expected state pay off, and how?

Two arms, the W7-X bands as published and the same bands scaled per volume by the
Fisher-information rule, each reconstructed on every catalog truth but
`EXCLUDED_TRUTHS`, once noise free and on `--n-seeds` noisy draws both arms share.

    python run_study.py --workers 5
    python run_study.py --arms fisher_bands --skip-stored --workers 5   # one arm only

Writes `inference_ids.json` beside this script and one record per run into the shared
`results_plots_tables/PlasmaStateInferenceResult/`.
"""

import argparse
import pathlib
import sys

_MODULE_DIRECTORY = pathlib.Path(__file__).resolve().parent
_PACKAGE_DIRECTORY = _MODULE_DIRECTORY.parents[3]
sys.path.insert(0, str(_PACKAGE_DIRECTORY.parent))

from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E402,E501
    desc_truth_types,
    inference_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E402,E501
    diagnostic_design_point,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E402,E501
    sweep_utils,
)

ARMS = {
    "w7x_bands": diagnostic_design_point.DiagnosticDesignPoint(
        band_scale_rule="none", band_scale=1.0
    ),
    "fisher_bands": diagnostic_design_point.DiagnosticDesignPoint(),
}
"""The two instruments compared, identical but for where the passband edges sit.

`w7x_bands` is the W7-X polychromator as published, every volume reading the same five
bands; `fisher_bands` is the pinned design point, each volume's bands scaled by the
Fisher-information rule at the state the reference equilibrium predicts there.
"""

METHOD: inference_types.InferenceMethod = "map_laplace"
"""The recipe every arm runs: the MAP with the Laplace approximation at it."""

PROFILE_FAMILY = "two_power_zero_edge"
"""The family every arm infers in, `center * (1 - rho^alpha)^beta` per profile."""

DEFAULT_N_SEEDS = 1
"""Noisy draws per cell, beside one noise-free cell."""

EXCLUDED_TRUTHS: tuple[str, ...] = ()
"""Catalog truths left out by default; none since the 2026-10-03 catalogue, which has no
pedestal truth, the one shape the two-power family could not represent."""

SEED_STRIDE = 1000
"""Seeds between consecutive truths, so no two truths ever share a draw."""

NOISE_FREE_SUFFIX = "noise_free"
"""Label suffix of the cell fitted to the expected signals."""

STUDY_NAME = _MODULE_DIRECTORY.name
"""What this study's records carry in `study_name`, and what its manifest selects on."""

IDS_FILE = _MODULE_DIRECTORY / "inference_ids.json"
"""One row per stored run, written by `sweep_utils.write_manifest` off the records."""


def main() -> None:
    arguments = _parse_arguments()
    paths = sweep_utils.paths_from_arguments(arguments, STUDY_NAME)
    stored = sweep_utils.current_ids(IDS_FILE) if arguments.skip_stored else {}
    cells = []
    catalog = sorted(desc_truth_types.DESC_TRUTH_IDS)
    for truth_name in arguments.truths:
        # Every arm reads the same realizations, so a difference between them is the
        # bands and not the draw.
        variants = [(NOISE_FREE_SUFFIX, arguments.seed, False)] + [
            (f"seed{seed}", seed, True)
            for seed in cell_seeds(
                arguments.seed, catalog.index(truth_name), arguments.n_seeds
            )
        ]
        for arm in arguments.arms:
            for suffix, seed, add_noise in variants:
                name = f"{truth_name}_{arm}_{suffix}"
                if name in stored:
                    continue
                cells.append(
                    sweep_utils.SweepCell(
                        name=name,
                        label=f"{truth_name:<16s} {arm:<13s} {suffix}",
                        truth_name=truth_name,
                        settings=_settings(seed, add_noise),
                        design_point=ARMS[arm],
                    )
                )
    print(
        f"[sweep] {len(cells)} cells: {len(arguments.arms)} band arms x "
        f"{len(arguments.truths)} truths x {arguments.n_seeds + 1} targets"
        + (f", {len(stored)} already stored and skipped" if stored else ""),
        flush=True,
    )
    sweep_utils.run_sweep(cells, paths, workers=arguments.workers)
    sweep_utils.write_manifest(IDS_FILE, STUDY_NAME)
    print(f"[sweep] manifest written to {IDS_FILE}")


def cell_seeds(base_seed: int, seed_index: int, n_seeds: int) -> list[int]:
    """The noise seeds of one truth's cells, disjoint from every other truth's.

    Args:
        base_seed: The study's `--seed`.
        seed_index: Position of the truth in the catalog.
        n_seeds: How many draws that truth gets.

    Returns:
        `n_seeds` consecutive seeds, offset by that position.
    """
    offset = base_seed + SEED_STRIDE * seed_index
    return [offset + replicate for replicate in range(n_seeds)]


def _settings(seed: int, add_noise: bool) -> inference_types.PlasmaEstimationSettings:
    """The inference configuration of one cell; only the target varies."""
    return inference_types.PlasmaEstimationSettings(
        method=METHOD,
        profile_family=PROFILE_FAMILY,
        seed=seed,
        add_measurement_noise=add_noise,
        verbose=False,
    )


def _parse_arguments() -> argparse.Namespace:
    """The study's command line; the arms themselves are fixed above."""
    defaults = inference_types.PlasmaEstimationSettings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--truths",
        nargs="+",
        default=[
            name
            for name in sorted(desc_truth_types.DESC_TRUTH_IDS)
            if name not in EXCLUDED_TRUTHS
        ],
        choices=sorted(desc_truth_types.DESC_TRUTH_IDS),
        help=(
            "Catalog ground truths every arm is run on; the default leaves out "
            "`EXCLUDED_TRUTHS`."
        ),
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        choices=list(ARMS),
        help="Which band sets to run; the default runs both.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help="Noisy draws per cell, beside the one noise-free cell.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Base seed; each truth's draws are `cell_seeds` of it.",
    )
    parser.add_argument(
        "--skip-stored",
        action="store_true",
        help="Skip cells the manifest already has a current run for.",
    )
    sweep_utils.add_sweep_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
