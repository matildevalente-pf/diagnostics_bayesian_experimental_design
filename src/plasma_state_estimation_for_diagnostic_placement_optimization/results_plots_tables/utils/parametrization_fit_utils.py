# Fitting a profile parametrization to a given profile, and scoring how well it can.
#
# Two studies need the same thing. The parametrization study asks how well each family
# can represent the training equilibria at all, which is the family's representation
# error and is measured with no inference anywhere. The method-comparison study needs
# the *target* parameter vector to normalize an inferred one by, and a truth is a
# `spline_knots` profile on the surrogate's grid that has no `two_power` parameters of
# its own, so the target is its projection onto the family being fitted.
#
# The fit minimizes the residual in the physical profile relative to its own peak.
#
# Lives on the analysis side because nothing in the inference fits a family to a known
# profile: the inference fits it to *data*. This is a diagnostic about the families.

import functools

import numpy as np
import scipy.optimize
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    profile_parametrization_types,
)

DEFAULT_N_RHO = 200
"""Radial points the fit residual is evaluated on."""

DEFAULT_MAX_RHO = 0.975
"""Outermost radius the fit is scored over.

Both families vanish at `rho = 1` by construction and so does the truth, so including
the last sliver would compare three zeros and flatter the fit.
"""

_TWO_POWER_BOUNDS = ((1e-9, 0.05, 0.01), (1e4, 500.0, 500.0))
"""Search box for `(center, alpha, beta)`, wide enough not to be the answer.

Deliberately far wider than the prior box of `priors_utils`: this measures what the
family can do, not what the inference is allowed to try.
"""


def fit_profile(
    family: profile_parametrization_types.ProfileFamily,
    target: np.ndarray,
    rho: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Best-fit parameters of one family against a given profile, and the error.

    Args:
        family: Which parametrization to fit.
        target: The profile to represent, in the family's own scaled units.
        rho: Radial grid `target` is sampled on.

    Returns:
        The fitted parameter vector and the root-mean-square residual as a percentage of
        the target's peak, which is the same normalization
        `inference_metrics_utils.ProfileAccuracy.error_percent` uses.

    Raises:
        ValueError: For a family that cannot be fitted to a profile this way.
    """
    if family != "two_power_zero_edge":
        raise ValueError(f"cannot fit {family!r} to a profile")
    solution = scipy.optimize.least_squares(
        functools.partial(_residual, family=family, target=target, rho=rho),
        np.array([float(target[0]), 2.0, 1.5]),
        bounds=_TWO_POWER_BOUNDS,
        max_nfev=20000,
    )
    return solution.x, error_percent(family, solution.x, target, rho)


def target_parameters(
    family: profile_parametrization_types.ProfileFamily,
    profiles: profile_parametrization_types.PlasmaProfiles,
    rho: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """The `ne` and `Te` parameter vectors a ground truth projects onto one family.

    The number an inferred parameter is normalized by, and the one a comparison table
    reports as the target. A truth is a `spline_knots` pair on the surrogate's grid, so
    it has no parameters of the fitted family at all and this projection is the only
    sense in which a "true parameter value" exists; its own residual, which
    `fit_profile` also returns, is the floor under any recovery measured against it.

    Both profiles are converted into the family's own units by
    `profile_parametrization_types.from_si` before fitting, which is the step that has
    to be got right and the reason that function exists: `two_power_zero_edge` is scaled
    by `DENSITY_UNIT` or `TEMPERATURE_UNIT` while `spline_knots` carries logs of SI
    directly. Dividing either way round by hand puts the parameters 19
    or 3 orders of magnitude off with no error anywhere.

    Args:
        family: Which parametrization to project onto.
        profiles: The ground truth's own `ne`/`Te` pair, in whatever family it carries.
        rho: Radial grid to fit over; `fit_grid()` when None.

    Returns:
        The `ne` and the `Te` parameter vector, in the target family's own units.
    """
    radius = fit_grid() if rho is None else np.asarray(rho)
    density = np.asarray(
        profile_parametrization_types.from_si(
            family,
            profile_parametrization_types.electron_density(profiles, radius),
            profile_parametrization_types.DENSITY_UNIT,
        )
    )
    temperature = np.asarray(
        profile_parametrization_types.from_si(
            family,
            profile_parametrization_types.electron_temperature(profiles, radius),
            profile_parametrization_types.TEMPERATURE_UNIT,
        )
    )
    return (
        fit_profile(family, density, radius)[0],
        fit_profile(family, temperature, radius)[0],
    )


def error_percent(
    family: profile_parametrization_types.ProfileFamily,
    params: np.ndarray,
    target: np.ndarray,
    rho: np.ndarray,
) -> float:
    """Root-mean-square residual of one parameter vector, as a percent of the peak.

    Args:
        family: Which parametrization `params` belongs to.
        params: The parameter vector to score.
        target: The profile it is meant to represent.
        rho: Radial grid both are evaluated on.

    Returns:
        The error in percent.
    """
    residual = np.asarray(_residual(params, family=family, target=target, rho=rho))
    return 100.0 * float(np.sqrt(np.mean(np.square(residual))))


def fit_grid(
    n_rho: int = DEFAULT_N_RHO, max_rho: float = DEFAULT_MAX_RHO
) -> np.ndarray:
    """The radial grid every fit in a study should share.

    Args:
        n_rho: Number of points.
        max_rho: Outermost radius.

    Returns:
        The grid, in `rho`.
    """
    return np.linspace(0.0, max_rho, n_rho)


def evaluate(
    family: profile_parametrization_types.ProfileFamily,
    params: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """One family evaluated on a radial grid, in its own scaled units.

    Args:
        family: Which parametrization `params` belongs to.
        params: The parameter vector.
        rho: Radial grid to evaluate on.

    Returns:
        The profile values.
    """
    return np.asarray(
        profile_parametrization_types.evaluate_profile(
            family,
            np.asarray(params),
            np.asarray(rho),
        )
    )


def _residual(
    params: np.ndarray,
    family: profile_parametrization_types.ProfileFamily,
    target: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """Fit residual, normalized by the target's peak so families share one scale."""
    fitted = evaluate(family, params, rho)
    return (fitted - target) / float(np.max(target))
