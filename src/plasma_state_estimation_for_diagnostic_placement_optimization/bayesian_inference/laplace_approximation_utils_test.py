# Tests of the compiled exact-Hessian Laplace fit on the real surrogate and one truth:
# 1. The Hessian is symmetric and matches central differences of the gradient of `U`.
#    Test: test_fit_hessian_is_symmetric_and_matches_central_differences
# 2. Its `J^T J` and prior precision equal the eagerly computed ones.
#    Test: test_fit_information_and_prior_precision_match_eager_computation
#
#   pytest src/.../bayesian_inference/laplace_approximation_utils_test.py -m integration

import functools
import pathlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    desc_truth_types,
    flat_posterior_utils,
    inference_model_utils,
    inference_target_types,
    inference_types,
    laplace_approximation_utils,
    map_optimization_utils,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    dataset_utils,
    model_storage_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.results_plots_tables.utils import (  # noqa: E501
    sweep_utils,
    synthetic_target_utils,
)

pytestmark = pytest.mark.integration

_DATA_DIRECTORY = pathlib.Path(__file__).resolve().parents[3] / "data"
_MODELS_DIRECTORY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "mhd_equilibrium_forward_models"
    / "trained_models"
)
_TRUTH = "wide_2_parametric"
_STEP = 1e-5


class _Fit(NamedTuple):
    settings: inference_types.PlasmaEstimationSettings
    priors: priors_utils.ProfilePriors
    target: inference_target_types.InferenceTarget
    surrogate: model_storage_utils.LoadedSurrogate
    posterior: flat_posterior_utils.FlatPosterior
    approximation: laplace_approximation_utils.LaplaceApproximation


@pytest.fixture(scope="module")
def fit() -> _Fit:
    """One MAP start on the default settings and the Laplace fit at its end point."""
    parquet_path = _DATA_DIRECTORY / f"{dataset_utils.DEFAULT_DATASET}.parquet"
    settings_path = _DATA_DIRECTORY / f"{dataset_utils.DEFAULT_DATASET}_settings.json"
    model_path = _MODELS_DIRECTORY / f"{inference_types.DEFAULT_SURROGATE_NAME}.npz"
    if not (parquet_path.exists() and model_path.exists()):
        pytest.skip(
            "needs the training parquet and the default surrogate; run "
            "mhd_equilibrium_forward_models/download_from_hugging_face.py"
        )
    settings = inference_types.PlasmaEstimationSettings(verbose=False)
    truth = desc_truth_types.read_desc_truth(_TRUTH, parquet_path, settings_path)
    surrogate = model_storage_utils.load_surrogate(
        _MODELS_DIRECTORY,
        inference_types.DEFAULT_SURROGATE_NAME,
        _DATA_DIRECTORY,
        machine_scale=truth.machine_scale,
    )
    placement = synthetic_target_utils.build_diagnostic_placement(
        diagnostic_design_point.DiagnosticDesignPoint(),
        sweep_utils.reference_truth(parquet_path, settings_path),
        verbose=False,
    )
    target = synthetic_target_utils.build_synthetic_target(
        truth,
        placement,
        interferometer_n_sample=settings.interferometer_n_sample,
        trim_chords_to_plasma=settings.trim_chords_to_plasma,
        seed=settings.seed,
        add_noise=settings.add_measurement_noise,
        interferometer_sigma=settings.interferometer_sigma,
    )
    priors = inference_model_utils.priors_for(settings, target.thomson_signals)
    posterior = next(
        map_optimization_utils.run_map(
            settings,
            priors,
            target,
            surrogate,
            inference_model_utils.start_points(settings, priors),
        )
    ).posterior
    approximation = laplace_approximation_utils.fit_laplace(
        posterior, settings, priors, target, surrogate
    )
    return _Fit(settings, priors, target, surrogate, posterior, approximation)


def test_fit_hessian_is_symmetric_and_matches_central_differences(fit: _Fit) -> None:
    hessian = np.asarray(fit.approximation.hessian)
    z = jnp.asarray(fit.approximation.mean)
    gradient = jax.jit(jax.grad(fit.posterior.potential))
    columns = []
    for index in range(z.shape[0]):
        direction = jnp.zeros_like(z).at[index].set(_STEP)
        columns.append(
            (gradient(z + direction) - gradient(z - direction)) / (2.0 * _STEP)
        )
    finite_difference = np.stack([np.asarray(column) for column in columns], axis=1)
    finite_difference = 0.5 * (finite_difference + finite_difference.T)

    np.testing.assert_array_equal(hessian, hessian.T)
    assert np.linalg.norm(hessian - finite_difference) / np.linalg.norm(hessian) < 1e-6


def test_fit_information_and_prior_precision_match_eager_computation(
    fit: _Fit,
) -> None:
    z = jnp.asarray(fit.approximation.mean)
    residuals = functools.partial(
        laplace_approximation_utils.residuals_of_z,
        posterior=fit.posterior,
        settings=fit.settings,
        target=fit.target,
        surrogate=fit.surrogate,
        flux_guess=inference_model_utils.precompute_flux_guess(
            fit.settings, fit.priors, fit.target, fit.surrogate
        ),
        fixed_coefficients=None,
    )
    jacobian = jax.jacrev(residuals)(z)
    precision = laplace_approximation_utils.prior_precision(
        z, fit.posterior, fit.priors
    )

    np.testing.assert_allclose(
        np.asarray(fit.approximation.information),
        np.asarray(jacobian.T @ jacobian),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(fit.approximation.prior_precision), np.asarray(precision), rtol=1e-12
    )
