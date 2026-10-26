# Convergence diagnostics of a finished NUTS run, rank-normalized as Vehtari et al.
# 2021 (https://doi.org/10.1214/20-BA1221) defines them, and the PASS or FAIL they give.

import typing
from collections.abc import Mapping
from typing import NamedTuple

import jaxtyping as jt
import numpy as np
import numpyro.diagnostics
import scipy.stats
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)

R_HAT_LIMIT = 1.01
"""Largest `r_hat` a converged run may show, on every sampled site."""

ESS_BULK_LIMIT = 400.0
"""Smallest bulk effective sample size a converged run may show.

Vehtari et al. 2021 recommend an average bulk ESS of at least 50 per split chain, so
four chains put the limit at 400, below which their `r_hat` and ESS estimates are
unreliable.
"""

ESS_TAIL_LIMIT = 400.0
"""Smallest tail effective sample size a converged run may show.

The band this work reports is the 5th to 95th percentile, which is a tail quantity, so
the tail count is the one that says whether the band itself is resolved.
"""

MAX_TREE_DEPTH_FRACTION_LIMIT = 0.01
"""Fraction of iterations that may saturate the doubling depth and still count as
essentially none. A saturating trajectory is cut before it turns back, so the sampler
is no longer the one whose stationary distribution is the posterior."""

TAIL_QUANTILES = (0.05, 0.95)
"""The quantiles the tail effective sample size is computed at, the edges of the
reported band."""


class SamplerDiagnostics(NamedTuple):
    """Everything that says whether a sampled posterior can be interpreted."""

    site_names: tuple[str, ...]
    """The sampled sites, in the order the per-site arrays below are in."""

    r_hat: jt.Float[np.ndarray, " n_site"]
    """Rank-normalized split `r_hat` per site, the larger of the plain and the folded
    statistic, which compares the variance between chains with the variance within
    them and sits at one for chains sampling the same distribution."""

    ess_bulk: jt.Float[np.ndarray, " n_site"]
    """Bulk effective sample size per site, the independent-draw count the chains are
    worth for a central quantity such as the median."""

    ess_tail: jt.Float[np.ndarray, " n_site"]
    """Tail effective sample size per site, the same count for the 5th and 95th
    percentiles, which is the quantity the reported band is built from."""

    r_hat_max: float
    """Worst `r_hat` over the sites, the number a run is judged by."""

    ess_bulk_min: float
    """Smallest bulk effective sample size over the sites."""

    ess_tail_min: float
    """Smallest tail effective sample size over the sites."""

    n_divergences: int
    """Divergent transitions over all chains, each one a trajectory the integrator
    could not follow, which biases the result wherever they cluster."""

    n_max_tree_depth: int
    """Iterations whose trajectory saturated `max_tree_depth` instead of ending on a
    U-turn."""

    max_tree_depth_fraction: float
    """`n_max_tree_depth` over the retained iterations."""

    accept_prob_mean: float
    """Mean Metropolis acceptance probability over the retained draws, which adaptation
    drives towards the configured target."""

    step_size_mean: float
    """Mean adapted leapfrog step size over the retained draws, constant per chain once
    warmup has ended."""

    ebfmi_min: float
    """Smallest energy Bayesian fraction of missing information over the chains.

    How well the momentum resampling explores the energy distribution, a value far
    below one meaning slow exploration of it rather than a failure of the diagnostics
    above, which live in the position space alone.
    """

    n_leapfrog_steps: int
    """Gradient evaluations of the retained iterations, the hardware-independent cost
    of the run."""

    n_draws: int
    """Retained draws per chain."""

    n_chains: int
    """Chains run."""

    converged: bool
    """Whether every criterion of this module is met, which is what says the posterior
    below may be read at all."""


def sampler_diagnostics(
    samples_by_chain: Mapping[str, jt.Float[NpOrJaxArray, "n_chain n_draw"]],
    extra_fields: Mapping[str, jt.Float[NpOrJaxArray, "n_chain n_draw"]],
    max_tree_depth: int,
) -> SamplerDiagnostics:
    """Every convergence diagnostic of one finished run, and the verdict they give.

    The per-site statistics are rank based and therefore unchanged by the monotone
    per-site transform between the constrained and the unconstrained space.

    Args:
        samples_by_chain: Retained draws per site, chain dimension first.
        extra_fields: The sampler's own per-draw fields, `diverging`, `num_steps`,
            `accept_prob`, `energy` and `adapt_state.step_size`, chain dimension first.
        max_tree_depth: The doubling cap the run was configured with, which fixes how
            many leapfrog steps a saturating iteration takes.

    Returns:
        The per-site and per-run diagnostics, with the PASS or FAIL in `converged`.
    """
    site_names = tuple(samples_by_chain)
    draws = {name: np.asarray(samples_by_chain[name]) for name in site_names}
    r_hat = np.array(
        [rank_normalized_split_r_hat(draws[name]) for name in site_names], dtype=float
    )
    ess_bulk = np.array(
        [bulk_effective_sample_size(draws[name]) for name in site_names], dtype=float
    )
    ess_tail = np.array(
        [tail_effective_sample_size(draws[name]) for name in site_names], dtype=float
    )
    num_steps = np.asarray(extra_fields["num_steps"])
    # A trajectory that never turns back doubles `max_tree_depth` times and therefore
    # takes `2^depth - 1` leapfrog steps, which is how a saturating iteration is read
    # off the step count numpyro records.
    saturating = num_steps >= 2**max_tree_depth - 1
    energy = np.asarray(extra_fields["energy"])
    n_chains, n_draws = num_steps.shape
    r_hat_max = _worst(r_hat, np.nanmax)
    ess_bulk_min = _worst(ess_bulk, np.nanmin)
    ess_tail_min = _worst(ess_tail, np.nanmin)
    converged = bool(
        r_hat_max < R_HAT_LIMIT
        and ess_bulk_min > ESS_BULK_LIMIT
        and ess_tail_min > ESS_TAIL_LIMIT
        and int(np.asarray(extra_fields["diverging"]).sum()) == 0
        and saturating.mean() <= MAX_TREE_DEPTH_FRACTION_LIMIT
    )
    return SamplerDiagnostics(
        site_names=site_names,
        r_hat=r_hat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        r_hat_max=r_hat_max,
        ess_bulk_min=ess_bulk_min,
        ess_tail_min=ess_tail_min,
        n_divergences=int(np.asarray(extra_fields["diverging"]).sum()),
        n_max_tree_depth=int(saturating.sum()),
        max_tree_depth_fraction=float(saturating.mean()),
        accept_prob_mean=float(np.mean(np.asarray(extra_fields["accept_prob"]))),
        step_size_mean=float(
            np.mean(np.asarray(extra_fields["adapt_state.step_size"]))
        ),
        ebfmi_min=float(np.min([energy_bfmi(row) for row in energy])),
        n_leapfrog_steps=int(num_steps.sum()),
        n_draws=int(n_draws),
        n_chains=int(n_chains),
        converged=converged,
    )


def rank_normalized_split_r_hat(
    draws: jt.Float[np.ndarray, "n_chain n_draw"],
) -> float:
    """Split `r_hat` of the rank-normalized draws, folded and unfolded.

    The folded statistic ranks the distance from the median instead of the value, so
    chains that agree on the location but not on the spread are caught as well.

    Args:
        draws: One site's retained draws, chain dimension first.

    Returns:
        The larger of the two statistics, nan for a single chain.
    """
    if draws.shape[0] < 2:
        return float("nan")
    folded = np.abs(draws - np.median(draws))
    return float(
        max(
            numpyro.diagnostics.split_gelman_rubin(_rank_normalize(draws)),
            numpyro.diagnostics.split_gelman_rubin(_rank_normalize(folded)),
        )
    )


def bulk_effective_sample_size(
    draws: jt.Float[np.ndarray, "n_chain n_draw"],
) -> float:
    """Effective sample size of the rank-normalized draws, on split chains.

    Args:
        draws: One site's retained draws, chain dimension first.

    Returns:
        The count, which for independent draws is the number of draws.
    """
    return float(
        numpyro.diagnostics.effective_sample_size(
            _split_chains(_rank_normalize(draws))
        )
    )


def tail_effective_sample_size(
    draws: jt.Float[np.ndarray, "n_chain n_draw"],
) -> float:
    """Smaller of the effective sample sizes of the two `TAIL_QUANTILES` indicators.

    Each quantile is estimated from the indicator series `draws <= quantile`, whose
    effective sample size is what the corresponding band edge is resolved with.

    Args:
        draws: One site's retained draws, chain dimension first.

    Returns:
        The count, nan where the draws are constant and the indicator carries no
        information.
    """
    counts = []
    for quantile in TAIL_QUANTILES:
        indicator = (draws <= np.quantile(draws, quantile)).astype(float)
        if np.all(indicator == indicator.flat[0]):
            return float("nan")
        counts.append(
            float(
                numpyro.diagnostics.effective_sample_size(_split_chains(indicator))
            )
        )
    return float(min(counts))


def energy_bfmi(energy: jt.Float[np.ndarray, " n_draw"]) -> float:
    """Energy Bayesian fraction of missing information of one chain.

    Args:
        energy: The Hamiltonian at every retained draw of that chain.

    Returns:
        `mean((E_n - E_{n-1})^2) / var(E)`, nan for a chain of constant energy.
    """
    variance = float(np.var(energy))
    if variance <= 0.0:
        return float("nan")
    return float(np.mean(np.square(np.diff(energy))) / variance)


def _worst(
    values: jt.Float[np.ndarray, " n_site"],
    reducer: typing.Callable[[jt.Float[np.ndarray, " n_site"]], np.floating],
) -> float:
    """The worst site's statistic, nan when no site has one at all.

    Reducing over sites directly would warn and still return nan on an all-nan slice,
    which is the single-chain case for `r_hat`.
    """
    if not np.any(np.isfinite(values)):
        return float("nan")
    return float(reducer(values))


def _rank_normalize(
    draws: jt.Float[np.ndarray, "n_chain n_draw"],
) -> jt.Float[np.ndarray, "n_chain n_draw"]:
    """Average ranks over all chains, put through the inverse normal transform.

    The offsets 3/8 and 1/4 are those of the Blom transform of Vehtari et al. 2021,
    eq. (14), which divides by `S + 1/4` where this divides by `S - 1/4`, a shift of the
    scores of order `1/S`. The transform makes the statistics below independent of the
    marginal's own shape.
    """
    n_total = draws.size
    ranks = scipy.stats.rankdata(draws, method="average").reshape(draws.shape)
    return scipy.stats.norm.ppf((ranks - 3.0 / 8.0) / (n_total - 1.0 / 4.0))


def _split_chains(
    draws: jt.Float[np.ndarray, "n_chain n_draw"],
) -> jt.Float[np.ndarray, "n_split_chain n_half_draw"]:
    """Each chain cut into its two halves, so a chain drifting within itself is seen."""
    half = draws.shape[1] // 2
    return np.concatenate([draws[:, :half], draws[:, -half:]], axis=0)
