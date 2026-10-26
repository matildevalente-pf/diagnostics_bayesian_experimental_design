# MHD surrogate quantities shared by several notebooks and scripts: the force residual
# and the `wout` scalars.

import numpy as np
import pandas as pd
import scipy.constants
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    evaluation_utils,
)

# Importing `desc` flips JAX to float64 globally, so every import of it stays inside the
# function that needs it.


GLOBAL_SCALARS = {
    "aspect": "R0/a",
    "volume": "V",
    "Aminor_p": "a",
    "Rmajor_p": "R0",
    "betatotal": "<beta>_vol",
    "betapol": "<beta_pol>_vol",
    "betator": "<beta_tor>_vol",
    "volavgB": "<|B|>_vol",
}
"""Global diagnostics of an equilibrium, by their `wout` name and their DESC name.

Named after the `wout` scalars a VMEC run reports, because that is what a reader of a
stellarator equilibrium expects to compare, and computed from DESC because this folder
holds no VMEC. `betaxis` and the boundary extents are not in this table: DESC has no
compute name for them, so `global_scalar_table` derives them itself.
"""


def normalized_force_residual(
    equilibrium: desc_types.DescEquilibrium,
    resolution: desc_types.DescResolution,
) -> float:
    """Compute DESC's own volume-averaged normalized force residual of an equilibrium.

    The same ratio the dataset records and the evaluation campaign reports, so a
    predicted equilibrium can be read against the 1e-2 the training set was filtered at.
    It is not the quantity the force-residual training term minimizes, which is
    normalized geometrically instead and lands on a different scale.

    Args:
        equilibrium: The equilibrium to score.
        resolution: Spectral resolution its coefficients are expressed in.

    Returns:
        `<|F|>_vol` divided by `<|grad(|B|^2)|/2mu0>_vol`.
    """
    return evaluation_utils.force_residual(
        desc_utils.desc_equilibrium_to_desc_object(equilibrium, resolution)
    )


def global_scalar_table(
    cases: dict[str, desc_types.DescEquilibrium],
    resolution: desc_types.DescResolution,
) -> pd.DataFrame:
    """Tabulate the global diagnostics of one or more equilibria side by side.

    The quantities a VMEC run reports as its `wout` scalars, under their `wout` names,
    so a predicted equilibrium can be read as a machine rather than as a coefficient
    vector. Every one of them follows from the geometry and the pressure alone, so none
    depends on the transform, which these equilibria do not prescribe.

    Args:
        cases: Column label to the equilibrium it holds.
        resolution: Spectral resolution the coefficients are expressed in.

    Returns:
        One row per scalar, one column per case, in the order of `GLOBAL_SCALARS`
        followed by `betaxis` and the boundary extents.
    """
    return pd.DataFrame(
        {
            label: _global_scalars(equilibrium, resolution)
            for label, equilibrium in cases.items()
        }
    )


def _global_scalars(
    equilibrium: desc_types.DescEquilibrium,
    resolution: desc_types.DescResolution,
) -> pd.Series:
    """Compute one equilibrium's column of `global_scalar_table`.

    Args:
        equilibrium: The equilibrium to evaluate.
        resolution: Spectral resolution its coefficients are expressed in.

    Returns:
        The scalars, indexed by their `wout` name.
    """
    import desc.grid

    live = desc_utils.desc_equilibrium_to_desc_object(equilibrium, resolution)
    volume = live.compute(list(GLOBAL_SCALARS.values()))
    scalars = {
        name: float(volume[quantity]) for name, quantity in GLOBAL_SCALARS.items()
    }
    surfaces = desc.grid.LinearGrid(
        rho=np.array([0.0, 1.0]),
        M=resolution.poloidal_resolution,
        N=resolution.toroidal_resolution,
        NFP=resolution.n_field_periods,
        sym=False,
    )
    on_surfaces = live.compute(["p", "|B|^2", "R", "Z"], grid=surfaces)
    axis = np.asarray(surfaces.nodes[:, 0]) == 0.0
    boundary = ~axis
    # VMEC's `betaxis`: the same ratio as `betatotal`, taken on the magnetic axis
    # rather than over the volume.
    scalars["betaxis"] = float(
        2.0
        * scipy.constants.mu_0
        * np.mean(np.asarray(on_surfaces["p"])[axis])
        / np.mean(np.asarray(on_surfaces["|B|^2"])[axis])
    )
    major_radius = np.asarray(on_surfaces["R"])[boundary]
    height = np.asarray(on_surfaces["Z"])[boundary]
    scalars["rmax_surf"] = float(major_radius.max())
    scalars["rmin_surf"] = float(major_radius.min())
    scalars["zmax_surf"] = float(height.max())
    return pd.Series(scalars)
