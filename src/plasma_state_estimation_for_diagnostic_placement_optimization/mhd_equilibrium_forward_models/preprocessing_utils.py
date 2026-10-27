import jax.numpy as jnp
import numpy as np
import sklearn.decomposition
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    preprocessing_types,
    surrogate_types,
)


def fit_preprocessor(
    x: Float[NpOrJaxArray, "n_samples n_rho"],
    y: Float[NpOrJaxArray, "n_samples n_y"],
    basis: surrogate_types.EquilibriumBasisSpecification,
    n_pca_components: int | None = None,
    energy_scale: float = 1.5,
) -> preprocessing_types.PreprocessorParams:
    """Fit the input and output normalization on the training rows.

    Args:
        x: Pressure profiles of the training rows [Pa].
        y: Flat targets of the training rows.
        basis: Basis of `y`, supplying the mode tables the energy scale is built from.
        n_pca_components: Number of PCA components to keep, or None to leave the target
            uncompressed. A count rather than a variance fraction, so the width does not
            depend on the reproducibility of the SVD.
        energy_scale: Decay length of the spectral weights; larger keeps more
            high-frequency content.

    Returns:
        The fitted parameters, with every array in float32.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    y_energy_scale = build_y_energy_scale(basis, energy_scale)
    y_scaled = y / y_energy_scale
    y_mean = y_scaled.mean(axis=0).astype(np.float32)
    y_std = np.where(y_scaled.std(axis=0) > 0, y_scaled.std(axis=0), 1.0).astype(
        np.float32
    )

    y_pca_components = None
    y_pca_mean = None
    if n_pca_components is not None:
        pca = sklearn.decomposition.PCA(n_components=n_pca_components)
        pca.fit(((y_scaled - y_mean) / y_std).astype(np.float32))
        y_pca_components = pca.components_.astype(np.float32)
        y_pca_mean = pca.mean_.astype(np.float32)

    return preprocessing_types.PreprocessorParams(
        x_mean=x.mean(axis=0).astype(np.float32),
        x_std=np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0).astype(np.float32),
        y_energy_scale=y_energy_scale,
        y_mean=y_mean,
        y_std=y_std,
        y_pca_components=y_pca_components,
        y_pca_mean=y_pca_mean,
    )


def build_y_energy_scale(
    basis: surrogate_types.EquilibriumBasisSpecification,
    energy_scale: float = 1.5,
) -> Float[NpOrJaxArray, " n_y"]:
    """Per-entry weight the flat target is divided by before z-scoring: each spectral
    coefficient gets `exp(-norm(l, m, n) / energy_scale)`, floored at 1e-6, and the
    iota block and `Psi` get one. The z-score cancels any per-entry weight, so the
    training target does not depend on `energy_scale`."""
    return np.concatenate(
        [
            _mode_weights(basis.r_modes, energy_scale),
            _mode_weights(basis.z_modes, energy_scale),
            _mode_weights(basis.lambda_modes, energy_scale),
            np.ones(basis.n_iota + 1, dtype=np.float32),
        ]
    )


def transform_x(
    params: preprocessing_types.PreprocessorParams,
    x: Float[NpOrJaxArray, "... n_rho"],
) -> Float[NpOrJaxArray, "... n_rho"]:
    """Z-score the pressure profiles into the network's input space.

    Args:
        params: Fitted normalization.
        x: One pressure profile or a batch of them [Pa].

    Returns:
        The normalized profiles, in the dtype of the fitted normalization.
    """
    return ((x - params.x_mean) / params.x_std).astype(params.x_mean.dtype)


def transform_y(
    params: preprocessing_types.PreprocessorParams,
    y: Float[NpOrJaxArray, "... n_y"],
) -> Float[NpOrJaxArray, "... n_latent"]:
    """Energy-scale, z-score and optionally PCA-project the flat targets.

    Args:
        params: Fitted normalization.
        y: One flat target or a batch of them.

    Returns:
        The targets in the space the network predicts in, in float32.
    """
    normalized = (((y / params.y_energy_scale) - params.y_mean) / params.y_std).astype(
        np.float32
    )
    if params.y_pca_components is None or params.y_pca_mean is None:
        return normalized
    return (normalized - params.y_pca_mean) @ params.y_pca_components.T


def inverse_transform_y(
    params: preprocessing_types.PreprocessorParams,
    latent: Float[NpOrJaxArray, "... n_latent"],
) -> Float[NpOrJaxArray, "... n_y"]:
    """Undo `transform_y`, back to the flat target space.

    Args:
        params: Fitted normalization.
        latent: One network output or a batch of them.

    Returns:
        The reconstructed flat targets, in the dtype of the fitted normalization.
    """
    normalized = latent
    if params.y_pca_components is not None and params.y_pca_mean is not None:
        normalized = latent @ params.y_pca_components + params.y_pca_mean
    dtype = params.y_mean.dtype
    scaled = (normalized * params.y_std + params.y_mean).astype(dtype)
    return (scaled * params.y_energy_scale).astype(dtype)


def fit_pressure_params_traceable(
    rho_grid: Float[NpOrJaxArray, " n_rho"],
    pressure_values: Float[NpOrJaxArray, " n_rho"],
    a_range: tuple[float, float] | None = None,
    b_range: tuple[float, float] | None = None,
    n_a_grid: int = 200,
) -> preprocessing_types.PressureParams:
    """Fit `p(rho) = p0 * (1 - rho^a)^b` to a sampled pressure profile, under a trace.

    `p0` is read off the axis, `b` follows in closed form from a weighted least squares in
    `log p` at each candidate `a`, and `a` is picked by scanning a grid. Every
    data-dependent branch of the original fit is a `jnp.where`, the masked quadrature is a
    set of trapezoidal weights zeroed at the invalid points, and the scan over `a`
    broadcasts rather than using `jax.vmap`, whose batching rules `jax2onnx` cannot lower.

    Because `a` comes from an `argmin`, the fit carries no gradient in `a`.

    Args:
        rho_grid: Radial points the profile is sampled on.
        pressure_values: Pressure at those points [Pa].
        a_range: Search interval for `a`, clipped into `[0.5, 10]`.
        b_range: Clamping interval for `b`, with a lower bound of at least 0.1.
        n_a_grid: Number of candidate values of `a`.

    Returns:
        The fitted parameters as JAX scalars.
    """
    rho_grid = jnp.asarray(rho_grid)
    pressure_values = jnp.asarray(pressure_values)
    dtype = jnp.result_type(pressure_values)
    p0 = pressure_values[0]

    a_lo, a_hi = (0.5, 10.0) if a_range is None else a_range
    a_lo = jnp.maximum(jnp.asarray(a_lo, dtype=dtype), 0.5)
    a_hi = jnp.minimum(jnp.asarray(a_hi, dtype=dtype), 10.0)
    b_lo, b_hi = (0.1, 1e6) if b_range is None else b_range
    b_lo = jnp.maximum(jnp.asarray(b_lo, dtype=dtype), 0.1)
    b_hi = jnp.asarray(b_hi, dtype=dtype)

    mask = (rho_grid > 0.01) & (rho_grid < 0.99) & (pressure_values > 0.0)
    spacing = jnp.concatenate(
        [
            rho_grid[1:2] - rho_grid[0:1],
            (rho_grid[2:] - rho_grid[:-2]) / 2.0,
            rho_grid[-1:] - rho_grid[-2:-1],
        ]
    )
    weights = jnp.where(mask, spacing, 0.0)

    normalized = pressure_values / jnp.maximum(p0, 1e-30)
    log_pressure = jnp.where(mask, jnp.log(jnp.maximum(normalized, 1e-30)), 0.0)

    candidates = jnp.linspace(a_lo, a_hi, n_a_grid)
    rho_pow_a = rho_grid[None, :] ** candidates[:, None]
    log_one_minus = jnp.log(jnp.maximum(1.0 - rho_pow_a, 1e-30))
    denominator = jnp.sum(weights[None, :] * log_one_minus**2, axis=1)
    numerator = jnp.sum(
        weights[None, :] * log_pressure[None, :] * log_one_minus, axis=1
    )
    b_candidates = jnp.clip(
        jnp.where(denominator > 1e-30, numerator / denominator, 1.0), b_lo, b_hi
    )
    residual = (
        normalized[None, :] - jnp.maximum(1.0 - rho_pow_a, 0.0) ** b_candidates[:, None]
    )
    best = jnp.argmin(jnp.sum(weights[None, :] * residual**2, axis=1))

    positive = p0 > 0.0
    return preprocessing_types.PressureParams(
        p0=jnp.where(positive, p0, jnp.asarray(0.0, dtype=dtype)),
        a=jnp.where(positive, candidates[best], jnp.asarray(2.0, dtype=dtype)),
        b=jnp.where(positive, b_candidates[best], jnp.asarray(1.0, dtype=dtype)),
    )


def _mode_weights(
    modes: Float[NpOrJaxArray, "n_modes 3"],
    energy_scale: float,
) -> Float[NpOrJaxArray, " n_modes"]:
    """Weight one block of spectral coefficients by its mode-number magnitude.

    Args:
        modes: `(l, m, n)` mode numbers of the block.
        energy_scale: Decay length of the weights.

    Returns:
        One positive weight per mode, in float32.
    """
    norm = np.sqrt(np.sum(np.asarray(modes, dtype=np.float64) ** 2, axis=1))
    return np.maximum(np.exp(-norm / energy_scale), 1e-6).astype(np.float32)
