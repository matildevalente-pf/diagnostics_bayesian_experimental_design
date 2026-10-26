import numpy as np
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    preprocessing_utils,
    testing_utils,
)


@pytest.mark.parametrize("n_pca_components", [None, 18])
def test_the_target_transform_round_trips(n_pca_components: int | None) -> None:
    """Uncompressed, and at full PCA rank, the transform is exactly invertible."""
    dataset = testing_utils.tiny_dataset(n_samples=64)
    assert dataset.basis.n_y == 18
    params = preprocessing_utils.fit_preprocessor(
        dataset.pressure, dataset.y, dataset.basis, n_pca_components=n_pca_components
    )
    y = np.asarray(dataset.y)[:4]

    recovered = preprocessing_utils.inverse_transform_y(
        params, preprocessing_utils.transform_y(params, y)
    )

    assert params.n_latent == 18
    np.testing.assert_allclose(recovered, y, rtol=0, atol=1e-4)


def test_the_energy_scale_decays_with_the_mode_number_and_spares_the_profiles() -> None:
    """High-frequency coefficients are boosted; the iota block and Psi are untouched."""
    basis = testing_utils.tiny_basis(n_iota=5)

    scale = preprocessing_utils.build_y_energy_scale(basis, energy_scale=1.5)

    assert scale.shape == (basis.n_y,)
    np.testing.assert_array_equal(scale[-(basis.n_iota + 1) :], 1.0)
    block = scale[: basis.n_r]
    assert np.all(np.diff(np.asarray(block)[[0, 2]]) < 0.0)
    assert np.all(scale > 0.0)
