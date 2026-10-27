# NUTS over the profile parameters, its chains started around the MAP, and the
# diagnostics that say whether its draws may be read; see the folder README.

import time
from collections.abc import Mapping
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import numpyro.infer
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_model_utils,
    inference_target_types,
    inference_types,
    mcmc_convergence_utils,
    posterior_utils,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    model_storage_utils,
)

EXTRA_FIELDS = (
    "diverging",
    "potential_energy",
    "energy",
    "num_steps",
    "accept_prob",
    "adapt_state.step_size",
)
"""Per-draw sampler state collected alongside the draws, which is what the convergence
diagnostics of `mcmc_convergence_utils` are computed from. Collected after warmup
alone, so every one of them describes the retained draws."""


class McmcResult(NamedTuple):
    """Posterior draws and the diagnostics that say whether to trust them."""

    unconstrained_samples: jt.Float[NpOrJaxArray, "n_sample n_param"]
    """The draws of every chain together, as flat unconstrained vectors in
    `posterior.site_names` order.

    The same coordinates the Laplace Gaussian lives in, so the two posteriors are
    comparable, and the form `band_from_unconstrained_draws` takes.
    """

    chain_samples: jt.Float[NpOrJaxArray, "n_chain n_sample n_param"]
    """The same draws in the constrained physical space, chain dimension kept.

    What a trace figure needs, and what the per-site diagnostics are computed from;
    those are rank based, so they are the same in either space.
    """

    warmup_chain_samples: jt.Float[NpOrJaxArray, "n_chain n_warmup n_param"]
    """The warmup draws in the layout of `chain_samples`, for the trace figure alone:
    they show the chains leaving their start points, and no diagnostic reads them."""

    log_posterior_trace: jt.Float[NpOrJaxArray, "n_chain n_sample"]
    """`-potential_energy` at every retained draw, one row per chain."""

    diagnostics: mcmc_convergence_utils.SamplerDiagnostics
    """Rank-normalized `r_hat`, the two effective sample sizes, the divergences, the
    tree-depth saturation and the PASS or FAIL they give."""

    seconds: float
    """Wall-clock time of warmup plus sampling, compilation included."""


def run_mcmc(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    initial_point: Mapping[str, jt.Float[NpOrJaxArray, ""]],
    posterior: posterior_utils.Posterior,
    laplace_covariance: jt.Float[NpOrJaxArray, "n_param n_param"],
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> McmcResult:
    """Run NUTS on the shared model, each chain started near the MAP but not on it.

    Every leapfrog step is one gradient of the log-posterior, hence one surrogate call
    and one inverse-map solve with its backward pass, so a run costs hours.

    Args:
        settings: The run configuration, supplying `settings.mcmc` and the seed.
        priors: Priors on both profiles.
        target: The observed data.
        surrogate: The equilibrium forward model.
        initial_point: The MAP point, which the chain start points are drawn around.
        posterior: The posterior, fixing the coordinate order of the draws.
        laplace_covariance: The Laplace covariance at the MAP in `z`, in the order of
            `posterior.site_names`, which the `laplace` starts are drawn from.
        fixed_coefficients: Geometry to hold frozen; None solves for it.

    Returns:
        The draws in both spaces, with the convergence diagnostics.
    """
    init_strategy = numpyro.infer.init_to_value(values=dict(initial_point))
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            inference_model_utils.bind_plasma_model(
                settings, priors, target, surrogate, fixed_coefficients
            ),
            init_strategy=init_strategy,
            max_tree_depth=settings.mcmc.max_tree_depth,
            target_accept_prob=settings.mcmc.target_accept_prob,
            dense_mass=settings.mcmc.dense_mass,
        ),
        num_warmup=settings.mcmc.n_warmup,
        num_samples=settings.mcmc.n_samples,
        num_chains=settings.mcmc.n_chains,
        # One CPU device, so the chains run one after another; the alternative maps
        # them over devices that are not there and falls back to this anyway.
        chain_method="sequential",
        progress_bar=False,
    )
    if settings.verbose:
        print(
            f"[MCMC] NUTS: {settings.mcmc.n_chains} chains, "
            f"{settings.mcmc.n_warmup} warmup, {settings.mcmc.n_samples} samples, "
            f"tree depth {settings.mcmc.max_tree_depth}, target accept "
            f"{settings.mcmc.target_accept_prob}, "
            f"{'dense' if settings.mcmc.dense_mass else 'diagonal'} mass matrix, "
            f"started at {settings.mcmc.init_strategy}",
            flush=True,
        )
    started = time.perf_counter()
    # Warmup apart from sampling, so its draws can be kept for the trace figure.
    mcmc.warmup(
        jax.random.PRNGKey(settings.seed + 1),
        collect_warmup=True,
        init_params=chain_initial_points(
            settings,
            initial_point,
            posterior,
            laplace_covariance,
            jax.random.PRNGKey(settings.seed + 3),
        ),
    )
    warmup_by_chain = mcmc.get_samples(group_by_chain=True)
    mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=EXTRA_FIELDS)
    seconds = time.perf_counter() - started
    by_chain = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    return McmcResult(
        unconstrained_samples=np.asarray(
            posterior_utils.unconstrain_samples(posterior, mcmc.get_samples())
        ),
        chain_samples=np.stack(
            [np.asarray(by_chain[name]) for name in posterior.site_names], axis=-1
        ),
        warmup_chain_samples=np.stack(
            [np.asarray(warmup_by_chain[name]) for name in posterior.site_names],
            axis=-1,
        ),
        log_posterior_trace=-np.asarray(extra_fields["potential_energy"]),
        diagnostics=mcmc_convergence_utils.sampler_diagnostics(
            {name: by_chain[name] for name in posterior.site_names},
            extra_fields,
            settings.mcmc.max_tree_depth,
        ),
        seconds=seconds,
    )


def chain_initial_points(
    settings: inference_types.PlasmaEstimationSettings,
    initial_point: Mapping[str, jt.Float[NpOrJaxArray, ""]],
    posterior: posterior_utils.Posterior,
    laplace_covariance: jt.Float[NpOrJaxArray, "n_param n_param"],
    key: jax.Array,
) -> dict[str, jt.Float[jax.Array, " n_chain"]]:
    """One start point per chain, drawn around the MAP in the unconstrained space.

    `laplace` draws from the Laplace Gaussian widened by `chain_init_width`, `map` adds
    `chain_init_jitter` noise; both in `z`, so every start satisfies the support.

    Args:
        settings: The run configuration, supplying the chain count and the spread.
        initial_point: The MAP point, per site, in the constrained space.
        posterior: The posterior, whose transforms carry the point into `z`.
        laplace_covariance: The Laplace covariance at the MAP, in the order of
            `posterior.site_names`; read by `laplace` alone.
        key: PRNG key of the perturbation, so the start points are reproducible.

    Returns:
        Per-site start values with a leading chain dimension, which is what NumPyro
        takes as `init_params`; unbatched for a single chain, as it requires.
    """
    names = posterior.site_names
    z_map = jnp.stack(
        [
            jnp.asarray(posterior.transforms[name].inv(jnp.asarray(initial_point[name])))
            for name in names
        ]
    )
    noise = jax.random.normal(key, (settings.mcmc.n_chains, len(names)))
    if settings.mcmc.init_strategy == "laplace":
        # Rows of `noise @ L^T`, `L` the Cholesky factor, are draws of the Gaussian.
        offsets = settings.mcmc.chain_init_width * (
            noise @ jnp.linalg.cholesky(jnp.asarray(laplace_covariance)).T
        )
    else:
        offsets = settings.mcmc.chain_init_jitter * noise
    z_chains = z_map + offsets
    if settings.mcmc.n_chains == 1:
        return {name: z_chains[0, index] for index, name in enumerate(names)}
    return {name: z_chains[:, index] for index, name in enumerate(names)}
