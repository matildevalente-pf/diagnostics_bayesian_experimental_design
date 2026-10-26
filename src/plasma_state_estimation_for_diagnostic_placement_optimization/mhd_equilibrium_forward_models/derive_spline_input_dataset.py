r"""Derive the dataset whose model input is the pressure at the stored spline knots.

The export writes each row's pressure resampled on a 20-point grid; the solver saw the
stored spline, and so do the force loss and the reported metric. This replaces the
`pressure` column by the spline's knot values and the settings' `pressure_rho_grid` by
the knots, leaving the rows, targets and pinned split untouched. Run it for the training
set and for the interpolator lattice, so both keep one basis:

    python src/plasma_state_estimation_for_diagnostic_placement_optimization/mhd_equilibrium_forward_models/derive_spline_input_dataset.py
    python src/plasma_state_estimation_for_diagnostic_placement_optimization/mhd_equilibrium_forward_models/derive_spline_input_dataset.py \\
        --source equilibrium_a_interpolator_28 \\
        --derived equilibrium_a_interpolator_28_spline_input
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

_PACKAGE_DIRECTORY = pathlib.Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source", default="equilibrium_a_3k", help="Exported dataset stem."
    )
    parser.add_argument(
        "--derived", default="equilibrium_a_3k_spline_input", help="Stem to write."
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=_PACKAGE_DIRECTORY.parents[1] / "data",
        help="Directory holding the source files; the derived ones go beside them.",
    )
    arguments = parser.parse_args()

    frame = pd.read_parquet(arguments.data_dir / f"{arguments.source}.parquet")
    knots = np.stack(list(frame["pressure_knots"]))
    if not np.all(knots == knots[0]):
        raise ValueError("The rows do not share one knot grid.")
    frame["pressure"] = frame["pressure_params"]
    frame.to_parquet(arguments.data_dir / f"{arguments.derived}.parquet", index=False)

    settings = json.loads(
        (arguments.data_dir / f"{arguments.source}_settings.json").read_text()
    )
    settings["dataset"] = arguments.derived
    settings["derived_from"] = arguments.source
    settings["pressure_rho_grid"] = knots[0].tolist()
    # The lattice names the training set it shares its basis with.
    if "training_dataset" in settings:
        settings["training_dataset"] = f"{settings['training_dataset']}_spline_input"
    (arguments.data_dir / f"{arguments.derived}_settings.json").write_text(
        json.dumps(settings, indent=4) + "\n"
    )
    if "data_split_id" in settings:
        write_split(arguments.data_dir / f"{arguments.derived}_split.json", settings)
    print(f"wrote {arguments.derived}: {len(frame)} rows, {knots.shape[1]} inputs")


def write_split(path: pathlib.Path, settings: dict[str, object]) -> None:
    """Write the pinned train/val/test split, the one part of the settings kept in git.

    Args:
        path: File to write.
        settings: The dataset settings carrying the split.
    """
    keys = ("data_split_id", "train_ids", "val_ids", "test_ids")
    # One line per key: the id lists run to thousands of entries, and a pretty-printed
    # array would put the whole dataset into a review diff.
    body = ",\n".join(
        f"    {json.dumps(key)}: {json.dumps(settings[key])}" for key in keys
    )
    path.write_text("{\n" + body + "\n}\n")


if __name__ == "__main__":
    main()
