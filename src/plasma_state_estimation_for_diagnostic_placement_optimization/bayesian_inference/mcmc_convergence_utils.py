# Convergence diagnostics of a finished NUTS run, rank-normalized as Vehtari et al.
# 2021 (https://doi.org/10.1214/20-BA1221) defines them, and the PASS or FAIL they give.

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
    r_hat, ess_bulk, ess_tail = [], [], []
    for name in site_names:
        draws = np.asarray(samples_by_chain[name])
        half = draws.shape[1] // 2
        # Rank normalization of the draws and of their distance from the median, with
        # the offsets 3/8 and 1/4 of the Blom transform (Vehtari et al. 2021, eq. 14).
        # The folded draws catch chains that agree on the location but not the spread.
        normalized = [
            scipy.stats.norm.ppf(
                (
                    scipy.stats.rankdata(values, method="average").reshape(values.shape)
                    - 3.0 / 8.0
                )
                / (values.size - 1.0 / 4.0)
            )
            for values in (draws, np.abs(draws - np.median(draws)))
        ]
        r_hat.append(
            max(numpyro.diagnostics.split_gelman_rubin(values) for values in normalized)
            if draws.shape[0] >= 2
            else float("nan")
        )
        # Every chain is split into its two halves, so a chain drifting within itself
        # is seen.
        ess_bulk.append(
            numpyro.diagnostics.effective_sample_size(
                np.concatenate([normalized[0][:, :half], normalized[0][:, -half:]])
            )
        )
        # Each band edge is estimated from the indicator `draws <= quantile`, whose
        # effective sample size says how well that edge is resolved; nan when an
        # indicator is constant and carries no information.
        indicators = [
            (draws <= np.quantile(draws, quantile)).astype(float)
            for quantile in TAIL_QUANTILES
        ]
        ess_tail.append(
            float("nan")
            if any(np.all(indicator == indicator.flat[0]) for indicator in indicators)
            else min(
                numpyro.diagnostics.effective_sample_size(
                    np.concatenate([indicator[:, :half], indicator[:, -half:]])
                )
                for indicator in indicators
            )
        )
    r_hat, ess_bulk, ess_tail = (
        np.array(values, dtype=float) for values in (r_hat, ess_bulk, ess_tail)
    )
    # The worst site decides; nan when no site has a finite value, the single-chain
    # case of `r_hat`, where `np.nanmax` would warn.
    r_hat_max = float(np.nanmax(r_hat)) if np.any(np.isfinite(r_hat)) else float("nan")
    ess_bulk_min = (
        float(np.nanmin(ess_bulk)) if np.any(np.isfinite(ess_bulk)) else float("nan")
    )
    ess_tail_min = (
        float(np.nanmin(ess_tail)) if np.any(np.isfinite(ess_tail)) else float("nan")
    )
    num_steps = np.asarray(extra_fields["num_steps"])
    # A trajectory that never turns back doubles `max_tree_depth` times and therefore
    # takes `2^depth - 1` leapfrog steps, which is how a saturating iteration is read
    # off the step count numpyro records.
    saturating = num_steps >= 2**max_tree_depth - 1
    n_divergences = int(np.asarray(extra_fields["diverging"]).sum())
    # Energy Bayesian fraction of missing information per chain,
    # `mean((E_n - E_{n-1})^2) / var(E)`, nan for a chain of constant energy.
    energy = np.asarray(extra_fields["energy"])
    energy_variance = np.var(energy, axis=1)
    ebfmi = np.where(
        energy_variance > 0.0,
        np.mean(np.square(np.diff(energy, axis=1)), axis=1)
        / np.where(energy_variance > 0.0, energy_variance, 1.0),
        np.nan,
    )
    n_chains, n_draws = num_steps.shape
    converged = bool(
        r_hat_max < R_HAT_LIMIT
        and ess_bulk_min > ESS_BULK_LIMIT
        and ess_tail_min > ESS_TAIL_LIMIT
        and n_divergences == 0
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
        n_divergences=n_divergences,
        n_max_tree_depth=int(saturating.sum()),
        max_tree_depth_fraction=float(saturating.mean()),
        accept_prob_mean=float(np.mean(np.asarray(extra_fields["accept_prob"]))),
        step_size_mean=float(
            np.mean(np.asarray(extra_fields["adapt_state.step_size"]))
        ),
        ebfmi_min=float(np.min(ebfmi)),
        n_leapfrog_steps=int(num_steps.sum()),
        n_draws=int(n_draws),
        n_chains=int(n_chains),
        converged=converged,
    )
