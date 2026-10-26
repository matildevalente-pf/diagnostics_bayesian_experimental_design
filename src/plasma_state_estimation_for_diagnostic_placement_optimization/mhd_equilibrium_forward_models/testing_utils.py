import numpy as np
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
    desc_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_types,
    surrogate_types,
)


def tiny_basis(
    n_rho: int = 6,
    n_iota: int = 5,
) -> surrogate_types.EquilibriumBasisSpecification:
    """Build a small basis specification on the real DESC mode tables.

    `L = 5, M = 1, N = 0, NFP = 1` is the resolution whose tables come out 6, 3 and 3
    modes wide, so `n_y` is 18 at the default `n_iota`.

    Args:
        n_rho: Number of pressure sample points, i.e. the model input width.
        n_iota: Number of points of the `iotaf` block of `y`.

    Returns:
        A basis specification whose `n_y` is `12 + n_iota + 1`.
    """
    resolution = desc_types.DescResolution(
        radial_resolution=5,
        poloidal_resolution=1,
        toroidal_resolution=0,
        n_field_periods=1,
    )
    r_modes, z_modes, lambda_modes = desc_utils.fourier_zernike_mode_tables(resolution)
    return surrogate_types.EquilibriumBasisSpecification(
        resolution=resolution,
        r_modes=r_modes,
        z_modes=z_modes,
        lambda_modes=lambda_modes,
        pressure_rho_grid=np.linspace(0.0, 1.0, n_rho),
        iota_rho_grid=np.sqrt(np.linspace(0.0, 1.0, n_iota)),
    )


def tiny_dataset(
    n_samples: int = 8,
    basis: surrogate_types.EquilibriumBasisSpecification | None = None,
    seed: int = 0,
    n_knots: int = 9,
) -> dataset_types.RawDataset:
    """Build a random dataset in `basis`, with the fnorm column spanning the 1% filter.

    The last two rows are given a force residual above 1% so that a filter applied at the
    default threshold is exercised by every caller.

    Args:
        n_samples: Number of rows, at least three.
        basis: Basis to express the rows in; a `tiny_basis()` when None.
        seed: Seed of the random number generator filling the arrays.
        n_knots: Number of knots of the stored pressure spline.

    Returns:
        A dataset whose rows are in a fixed pseudo-random order.
    """
    # Two rows are forced above the filter, so fewer than three would leave none below it.
    if n_samples < 3:
        raise ValueError(f"n_samples must be at least 3, got {n_samples}.")
    basis = tiny_basis() if basis is None else basis
    generator = np.random.default_rng(seed)
    fnorm = generator.uniform(1e-4, 5e-3, size=n_samples)
    fnorm[-2:] = [2e-2, 5e-2]
    return dataset_types.RawDataset(
        basis=basis,
        pressure=generator.uniform(0.0, 3.5e4, size=(n_samples, basis.n_rho)),
        y=generator.normal(size=(n_samples, basis.n_y)),
        pressure_knots=np.tile(np.linspace(0.0, 1.0, n_knots), (n_samples, 1)),
        pressure_params=generator.uniform(0.0, 3.5e4, size=(n_samples, n_knots)),
        fnorm_vol=fnorm,
        beta_vol=generator.uniform(0.0, 0.06, size=n_samples),
        equilibrium_ids=[f"D{index:022d}" for index in range(n_samples)],
    )
