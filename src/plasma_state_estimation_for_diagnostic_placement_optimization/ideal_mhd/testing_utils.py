import desc.equilibrium
import desc.geometry
import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)


def coefficients_from_equilibrium(
    equilibrium: desc.equilibrium.Equilibrium,
) -> desc_types.DescFourierZernikeCoefficients:
    """Extract the Fourier-Zernike geometry coefficients of a live DESC equilibrium.

    Args:
        equilibrium: A `desc.equilibrium.Equilibrium`.

    Returns:
        The geometry coefficients as `DescFourierZernikeCoefficients`.
    """
    return desc_types.DescFourierZernikeCoefficients(
        r_lmn=np.asarray(equilibrium.R_lmn, dtype=float),
        z_lmn=np.asarray(equilibrium.Z_lmn, dtype=float),
        lambda_lmn=np.asarray(equilibrium.L_lmn, dtype=float),
        r_modes=np.asarray(equilibrium.R_basis.modes, dtype=int),
        z_modes=np.asarray(equilibrium.Z_basis.modes, dtype=int),
        lambda_modes=np.asarray(equilibrium.L_basis.modes, dtype=int),
        n_field_periods=int(equilibrium.NFP),
    )


def rotating_ellipse_equilibrium() -> desc.equilibrium.Equilibrium:
    """A rotating-ellipse stellarator equilibrium (`NFP = 5`), built in-process.

    The default `desc.Equilibrium()` is an axisymmetric circular torus whose `(R, Z)`
    are linear in `(rho cos(theta), rho sin(theta))`, so Newton inverts it exactly in
    one step from any guess; the rotating ellipse makes the inversion genuinely
    nonlinear. It is the zero-data offline geometry of the inverse-map tests and the
    stress soak.

    Returns:
        The `Equilibrium` (`L=8, M=6, N=2, NFP=5, sym=True`).
    """
    surface = desc.geometry.FourierRZToroidalSurface(
        R_lmn=np.array([10.0, 1.0, 0.3]),
        modes_R=np.array([[0, 0], [1, 0], [1, 1]]),
        Z_lmn=np.array([-1.0, -0.3]),
        modes_Z=np.array([[-1, 0], [-1, 1]]),
        NFP=5,
    )
    return desc.equilibrium.Equilibrium(L=8, M=6, N=2, NFP=5, sym=True, surface=surface)
