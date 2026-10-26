# The prior on the profile parameters, one NumPyro distribution per sample site: a
# measured uniform box for `two_power_zero_edge`, a log-uniform box per calibration.

from collections.abc import Iterable, Mapping

import numpy as np
import numpyro.distributions
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    thomson_scattering_utils,
)

ProfilePriors = dict[str, numpyro.distributions.Distribution]
"""One prior per sample site, keyed `ne_<index>` and `te_<index>` in parameter order,
plus one `CALIBRATION_PREFIX_<volume index>` per volume whose Thomson calibration
correction is inferred.

The distribution is the whole prior: it samples, it scores, its `mean` is where a
profile site starts from, and `len` is the parameter count.
"""

CALIBRATION_PREFIX = "calibration"
"""Prefix of the sites holding the Thomson calibration corrections `delta_j`, one per
inferred scattering volume and suffixed by that volume's index."""


def default_priors(
    profile_family: profile_parametrization_types.ProfileFamily,
    calibration_volume_indices: Iterable[int] = (),
    calibration_bound: float = thomson_scattering_utils.CALIBRATION_BOUND,
) -> ProfilePriors:
    """The priors of one family; only the inferable family has any.

    Args:
        profile_family: Which parametrization the priors describe.
        calibration_volume_indices: Scattering volumes whose Thomson calibration
            correction is inferred; empty leaves every calibration known.
        calibration_bound: `N` of the box `[1/N, N]` each of their corrections keeps to.

    Returns:
        The priors of that family, plus one calibration site per named volume.

    Raises:
        ValueError: For a family this folder does not infer.
    """
    if profile_family != "two_power_zero_edge":
        raise ValueError(
            f"{profile_family} is not an inferable family and has no prior. Use "
            f"two_power_zero_edge."
        )
    priors = two_power_priors()
    priors.update(calibration_priors(calibration_volume_indices, calibration_bound))
    return priors


def calibration_priors(
    volume_indices: Iterable[int],
    bound: float = thomson_scattering_utils.CALIBRATION_BOUND,
) -> ProfilePriors:
    """`ln delta_j ~ U(-ln N, ln N)`, a log-uniform box on each volume's Thomson
    calibration correction, symmetric under `delta_j -> 1 / delta_j` because the
    correction is a factor: a third and the triple are equally likely.

    Thomson constrains the product of `delta_j` and the local density, so the absolute
    density of the interferometer is what separates the two, as it separates the single
    global factor of Merlo 2023, section 2.2,
    https://doi.org/10.1088/1748-0221/18/11/P11012.

    Args:
        volume_indices: Scattering volumes to infer a correction for.
        bound: `N`, the edge of the box, shared by every volume.

    Returns:
        One prior per volume, keyed `calibration_<volume index>`.
    """
    # `validate_args=True` for the same reason as the profile boxes below.
    return {
        f"{CALIBRATION_PREFIX}_{int(index)}": numpyro.distributions.LogUniform(
            1.0 / bound, bound, validate_args=True
        )
        for index in volume_indices
    }


def two_power_priors() -> ProfilePriors:
    """Uniform boxes on `[center, alpha, beta]` of both profiles, in scaled units: axis
    values capped just above the dataset's largest, `alpha` wide enough for the prior's
    profile band to cover the broadest training profiles, `beta` around their bulk.

    Returns:
        Three priors per profile, in `TWO_POWER_ZERO_EDGE_PARAM_NAMES` order.
    """
    # The caps sit about a quarter above the largest axis value of either split at the
    # machine scale (`diagnostic_design_point.MACHINE_SCALE`) under the split
    # anchored at `diagnostic_design_point.TE_REFERENCE_EV`; `alpha` below one is a cusp
    # on the axis, and up to 60 the prior's 5-95% band in profile space covers the
    # broadest training profiles inside rho = 0.97.
    boxes = {
        "ne_0": (0.2, 95.0),  # center
        "ne_1": (1.0, 60.0),  # alpha
        "ne_2": (0.05, 12.0),  # beta
        "te_0": (0.04, 15.0),
        "te_1": (1.0, 60.0),
        "te_2": (0.05, 12.0),
    }
    # `validate_args=True` is not decoration: NumPyro defaults it off, and then
    # `Uniform.log_prob` returns its in-support constant far outside the box.
    return {
        name: numpyro.distributions.Uniform(low, high, validate_args=True)
        for name, (low, high) in boxes.items()
    }


def site_names(names: Iterable[str], prefix: str) -> list[str]:
    """One profile's site names, in index order rather than in dictionary order.

    `sorted` on the names themselves would put `ne_10` before `ne_2`, the same trap
    `flat_posterior_utils.site_names` measures rather than assumes.

    Args:
        names: The priors, a point, or anything else keyed by site name.
        prefix: `ne`, `te`, or `CALIBRATION_PREFIX` for the calibration corrections.

    Returns:
        The names of that profile's sites, ordered by index.
    """
    indices = sorted(
        int(name.split("_")[1]) for name in names if name.startswith(f"{prefix}_")
    )
    return [f"{prefix}_{index}" for index in indices]


def stack_profile_columns(
    columns: Mapping[str, NpOrJaxArray],
    priors: ProfilePriors,
    prefix: str,
) -> np.ndarray:
    """One profile's per-site draws as a `(n_sample, n_param)` matrix, in index order.

    Args:
        columns: Draws keyed by site name.
        priors: The priors whose sites name the columns.
        prefix: `ne` or `te`, selecting the profile.

    Returns:
        The stacked draws.
    """
    return np.stack(
        [np.asarray(columns[name]) for name in site_names(priors, prefix)], axis=-1
    )
