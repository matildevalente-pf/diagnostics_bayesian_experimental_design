r"""Regenerate one dataset row with DESC and compare it with the stored row.

Every row of both datasets is one DESC force-balance solve from the same starting point,
a vacuum equilibrium computed once with VMEC (fixed boundary), converted to DESC and
solved again. This script rebuilds that start from `equilibrium_a_warm_start.json`, sets
the row's pressure spline, a zero current on the same knots and the row's `Psi`, solves
as the dataset generation did, and prints how far the result is from the stored
coefficients. One solve takes about 7 min and 10 GB of memory.

    cd src/plasma_state_estimation_for_diagnostic_placement_optimization
    python mhd_equilibrium_forward_models/regenerate_dataset_row.py \\
        --row-id DgSADUTGs7s2ZBxE5xge2ej
"""

import argparse
import json
import logging
import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_utils,
    download_from_hugging_face,
    surrogate_types,
)

if TYPE_CHECKING:
    import desc.equilibrium


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--row-id", required=True, help="`desc_equilibrium_id` of the row to solve."
    )
    parser.add_argument(
        "--dataset",
        default=dataset_utils.DEFAULT_DATASET,
        help="Stem of the dataset parquet and its settings file.",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=download_from_hugging_face.DATA_DIRECTORY,
        help="Directory of the dataset and the starting point; missing files are "
        "downloaded.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    download_from_hugging_face.logger.setLevel(logging.INFO)

    parquet = arguments.data_dir / f"{arguments.dataset}.parquet"
    settings = arguments.data_dir / f"{arguments.dataset}_settings.json"
    warm_start = arguments.data_dir / download_from_hugging_face.WARM_START
    for path in (parquet, settings, warm_start):
        download_from_hugging_face.download(
            download_from_hugging_face.DATASETS_REPOSITORY, path.name, path
        )
    row = read_row(parquet, arguments.row_id)
    equilibrium = starting_equilibrium(
        json.loads(warm_start.read_text()),
        row,
        dataset_utils.load_metadata(settings).basis,
    )
    # The call the dataset generation made, at DESC's default optimizer and tolerances.
    equilibrium.solve(objective="force", verbose=3)

    for name, solved, stored in (
        ("R_lmn", equilibrium.R_lmn, row["R_lmn"]),
        ("Z_lmn", equilibrium.Z_lmn, row["Z_lmn"]),
        ("lambda_lmn", equilibrium.L_lmn, row["lambda_lmn"]),
    ):
        difference = np.abs(np.asarray(solved) - np.asarray(stored, dtype=float))
        print(f"{name}: max |regenerated - stored| = {difference.max():.3e}")


def read_row(parquet: pathlib.Path, row_id: str) -> pd.Series:
    """The row of `parquet` whose `desc_equilibrium_id` is `row_id`."""
    rows = pd.read_parquet(parquet, filters=[("desc_equilibrium_id", "==", row_id)])
    if len(rows) != 1:
        raise KeyError(f"{row_id} is not a row of {parquet.name}")
    return rows.iloc[0]


def starting_equilibrium(
    warm_start: dict[str, Any],
    row: pd.Series,
    basis: surrogate_types.EquilibriumBasisSpecification,
) -> "desc.equilibrium.Equilibrium":
    """The DESC equilibrium a row was solved from: the shared start with the row's
    pressure spline, a zero current on its knots and its `Psi`."""
    knots = np.asarray(row["pressure_knots"], dtype=float)
    method = warm_start["spline_method"]
    equilibrium = desc_utils.desc_equilibrium_to_desc_object(
        desc_types.DescEquilibrium(
            coefficients=desc_types.DescFourierZernikeCoefficients(
                r_lmn=np.asarray(warm_start["R_lmn"]),
                z_lmn=np.asarray(warm_start["Z_lmn"]),
                lambda_lmn=np.asarray(warm_start["lambda_lmn"]),
                r_modes=basis.r_modes,
                z_modes=basis.z_modes,
                lambda_modes=basis.lambda_modes,
                n_field_periods=basis.resolution.n_field_periods,
            ),
            total_toroidal_flux=float(row["Psi"]),
            pressure_profile=desc_types.DescSplineProfile(
                name="pressure",
                knots=knots,
                params=np.asarray(row["pressure_params"], dtype=float),
                method=method,
            ),
            current_profile=desc_types.DescSplineProfile(
                name="current", knots=knots, params=np.zeros_like(knots), method=method
            ),
        ),
        basis.resolution,
    )
    # Built without a surface, DESC holds its default torus as boundary and axis. The
    # generation overwrote both with these, and the fixed-boundary solve is constrained
    # by the boundary, so without them it would solve for another configuration.
    equilibrium.Ra_n = np.asarray(warm_start["Ra_n"])
    equilibrium.Za_n = np.asarray(warm_start["Za_n"])
    equilibrium.Rb_lmn = np.asarray(warm_start["Rb_lmn"])
    equilibrium.Zb_lmn = np.asarray(warm_start["Zb_lmn"])
    return equilibrium


if __name__ == "__main__":
    main()
