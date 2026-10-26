import pathlib

import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)


def load_equilibrium_npz(
    path: str | pathlib.Path,
) -> desc_types.DescEquilibrium:
    """Load a DESC equilibrium, geometry and profiles, from a local `.npz` artifact.

    Every profile the artifact carries is picked up from its `<name>_knots` /
    `<name>_params` pair, so an equilibrium stored with a current profile and one
    stored with an iota profile both load without a flag.

    Args:
        path: Path of an `.npz` with the coefficient arrays, the total toroidal flux
            and the profile knots/params (see the data READMEs).

    Returns:
        The equilibrium.
    """
    with np.load(path) as data:
        profiles = {
            f"{name}_profile": desc_types.DescSplineProfile(
                name=name, knots=data[f"{name}_knots"], params=data[f"{name}_params"]
            )
            for name in (
                "pressure",
                "iota",
                "current",
                "electron_temperature",
                "electron_density",
            )
            if f"{name}_knots" in data
        }
        return desc_types.DescEquilibrium(
            coefficients=desc_types.DescFourierZernikeCoefficients(
                r_lmn=data["r_lmn"],
                z_lmn=data["z_lmn"],
                lambda_lmn=data["lambda_lmn"],
                r_modes=data["r_modes"],
                z_modes=data["z_modes"],
                lambda_modes=data["lambda_modes"],
                n_field_periods=int(data["n_field_periods"]),
            ),
            total_toroidal_flux=float(data["total_toroidal_flux"]),
            **profiles,
        )
