# How far a finished run landed from the truth it was generated from: what the figures
# and tables read off it.

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_types,
)

PROFILE_NAMES = ("electron_density", "electron_temperature", "pressure")
"""The three curves every posterior reports, in the order they are summarized."""


class ProfileAccuracy(NamedTuple):
    """How far one inferred profile is from the truth, and how wide it claims to be.

    Reported together on purpose: a band is only meaningful next to the error it is
    supposed to bracket, and `truth_coverage_fraction` says whether it brackets it.
    """

    name: str
    """Which of `PROFILE_NAMES` this describes."""

    error_percent: float
    """Root-mean-square deviation from the truth, as a percentage of the truth's peak.

    Normalized by the peak rather than pointwise, because both families vanish at the
    boundary and a pointwise relative error diverges there however good the fit is.
    """

    axis_error_percent: float
    """Pointwise relative deviation at `rho = 0`, where the profile is largest."""

    uncertainty_percent: float
    """Median half-width of the reported band, as a percentage of the local value.

    The band is multiplicative, so the half-width relative to the reported mean is
    finite wherever the mean is, and the median over `rho` is a fair single number.
    """

    truth_coverage_fraction: float
    """Fraction of the radial grid where the reported band contains the truth.

    One for a band honest about its own error, zero for one far narrower than the error
    it should have bracketed. This decides whether an uncertainty can be quoted at all.
    """


def profile_accuracy(
    result: inference_types.PlasmaStateInferenceResult,
) -> tuple[ProfileAccuracy, ...]:
    """Summarize a finished run against the truth it was generated from.

    Reads the record alone, so a notebook gets the same numbers the runner printed.

    Args:
        result: A stored or freshly built inference record.

    Returns:
        One summary per entry of `PROFILE_NAMES`.
    """
    inferred, truth = result.inferred_profiles, result.true_profiles
    summaries = []
    for name in PROFILE_NAMES:
        central = central_curve(inferred, name)
        lower = np.asarray(getattr(inferred, f"{name}_lower"), dtype=float)
        upper = np.asarray(getattr(inferred, f"{name}_upper"), dtype=float)
        true = np.asarray(getattr(truth, f"{name}_median"), dtype=float)
        peak = float(np.max(np.abs(true)))
        positive = central > 0.0
        half_width = np.where(
            positive,
            0.5 * (upper - lower) / np.where(positive, central, 1.0),
            np.nan,
        )
        summaries.append(
            ProfileAccuracy(
                name=name,
                error_percent=(
                    100.0 * float(np.sqrt(np.mean(np.square(central - true)))) / peak
                    if peak > 0.0
                    else float("nan")
                ),
                axis_error_percent=(
                    100.0 * abs(float(central[0] - true[0])) / abs(float(true[0]))
                    if true[0] != 0.0
                    else float("nan")
                ),
                uncertainty_percent=100.0 * float(np.nanmedian(half_width)),
                truth_coverage_fraction=float(
                    np.mean((true >= lower) & (true <= upper))
                ),
            )
        )
    return tuple(summaries)


def central_curve(
    posterior: inference_types.PlasmaProfilePosterior, name: str
) -> np.ndarray:
    """The reported profile of a band: its mean.

    Args:
        posterior: The band to read.
        name: One of `PROFILE_NAMES`.

    Returns:
        The curve on the band's own `rho`.
    """
    return np.asarray(getattr(posterior, f"{name}_mean"), dtype=float)


def sweep_metrics_table(
    records: Iterable[inference_types.PlasmaStateInferenceResult],
) -> pd.DataFrame:
    """One row per run and profile: what the run read, how far it landed, how wide.

    The long form of `profile_accuracy` over many runs, with the fields a sweep groups
    on. Reads the records alone, so a grid of hundreds of runs costs no array loads.

    Args:
        records: The stored runs to summarize, in any order.

    Returns:
        A row per `(run, profile)`, the diagnostic counts those the run read: its
        masked subset, or every one its layout built.
    """
    rows = []
    for record in records:
        shared = {
            "inference_id": record.inference_id,
            "study_label": record.study_label,
            "truth_name": record.truth_name,
            "method": record.settings.method,
            "profile_family": record.settings.profile_family,
            "seed": record.settings.seed,
            "n_thomson_volumes": _count(
                record.thomson_volume_indices,
                len(record.design_point.volume_span_fractions),
            ),
            "n_interferometer_chords": _count(
                record.interferometer_chord_indices,
                len(record.design_point.chord_surface_angles),
            ),
            "information_gain_bits": record.information_gain_bits,
            "total_time_seconds": record.total_time_seconds,
        }
        for accuracy in profile_accuracy(record):
            rows.append({**shared, **accuracy._asdict()})
    return pd.DataFrame(rows)


def arm_comparison_table(
    metrics: pd.DataFrame,
    profile_name: str,
    arms: tuple[str, ...],
    value_columns: tuple[str, ...] = ("uncertainty_percent", "error_percent"),
) -> pd.DataFrame:
    """One row per truth, one column block per arm, and a mean and spread row below.

    Every entry is the mean over that truth's noisy draws; the last two rows are the
    mean over the truths and the standard deviation between them.

    Args:
        metrics: The long table `sweep_metrics_table` returns, carrying an `arm` column
            that names which arm each run belongs to.
        profile_name: Which of `PROFILE_NAMES` the table reports.
        arms: The arms, in the order their column blocks appear.
        value_columns: What each arm reports, in the order the columns appear.

    Returns:
        A table indexed by truth name with a `mean` and a `standard_deviation` row.
    """
    selected = metrics[metrics["name"] == profile_name]
    per_truth = selected.pivot_table(
        index="truth_name",
        columns="arm",
        values=list(value_columns),
        aggfunc="mean",
    ).reorder_levels([1, 0], axis=1)
    per_truth = per_truth.reindex(
        columns=pd.MultiIndex.from_product([arms, value_columns])
    )
    summary = pd.DataFrame(
        {"mean": per_truth.mean(), "standard_deviation": per_truth.std()}
    ).T
    # The index is left unnamed: with a two-level column header pandas writes an index
    # name on a third header line, which no reader of the CSV wants.
    return pd.concat([per_truth, summary])


def _count(indices: list[int] | None, n_built: int) -> int:
    """How many diagnostics a run read: its kept indices, or all `n_built` unmasked."""
    return n_built if indices is None else len(indices)
