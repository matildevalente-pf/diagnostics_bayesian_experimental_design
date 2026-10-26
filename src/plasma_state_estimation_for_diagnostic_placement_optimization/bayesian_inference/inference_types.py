# The configuration of one inference run, and the record it produces.

from typing import Annotated, Literal

import jaxtyping as jt
import pydantic
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
    array_serializer,
    array_validator,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    profile_parametrization_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.diagnostics_forward_models import (  # noqa: E501
    diagnostic_design_point,
    interferometer_utils,
    thomson_scattering_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    surrogate_types,
)

DEFAULT_SURROGATE_NAME = "20260907_081512_mlp_data_fnorm_only_pca_hand_tuned"
"""The equilibrium surrogate every study in this area consumes, one name in one place so
studies cannot silently compare results from different forward models: the best model of
the test-set comparison, reading the pressure at the 99 spline knots."""

DEFAULT_N_MAP_STARTS = 1
"""MAP starts of every run, each a draw from the priors keyed by the noise seed.

As in Merlo et al. (2023, JINST 18 P11012) (author's decision, 2026-10-02, replacing the
ten starts of 2026-09-29); with more, the converged one with the lowest `U` is reported.
"""

ChainInitStrategy = Literal["laplace", "map"]
"""Where a sampled run starts its chains; see `McmcSettings.init_strategy`."""

InferenceMethod = Literal["map_laplace", "mcmc_nuts"]
"""One complete recipe: the MAP, then where its credible band comes from.
`map_laplace` expands the posterior to second order at the MAP (exact Hessian);
`mcmc_nuts` samples the posterior itself and assumes no parametric form."""


class MapSettings(pydantic.BaseModel):
    """Settings of the MAP optimization every recipe starts from: L-BFGS with a zoom
    line search on the negative log posterior in the unconstrained space."""

    model_config = pydantic.ConfigDict(frozen=True)

    max_iterations: Annotated[int, pydantic.Field(ge=1)] = 1_000
    """Ceiling on L-BFGS iterations per start; a start that reaches it is not
    converged."""

    objective_tolerance: Annotated[float, pydantic.Field(gt=0.0)] = 1e-4
    """A start stops once an accepted step on a freshly reset L-BFGS memory lowers
    `U(z)` by less than this many nats, or that line search fails; with curvature pairs
    in memory such a step, like a failed line search, resets them."""

    decrement_tolerance: Annotated[float, pydantic.Field(gt=0.0)] = 1.0
    """A start that stopped before the cap has converged when its curvature is positive
    definite and its Newton decrement `g^T H^-1 g` is below this, about half a nat."""

    reduced_chi_squared_tolerance: Annotated[float, pydantic.Field(gt=0.0)] = 3.0
    """A converged start must also explain its data, its residuals within their
    uncertainties (Fischer et al. 2024, Sec. 2.8), tested as `chi^2 / dof` below this
    threshold, a choice of this work; a target without degrees of freedom skips it."""

    max_attempts: Annotated[int, pydantic.Field(ge=1)] = 3
    """Prior draws a run tries in turn, past its `n_starts`, until one converges; then a
    warm start if the run was given one, and with none converged the best attempt."""

    warm_start_first: bool = False
    """Try the warm start before the prior draws instead of after them, so a run
    restarted from another run's MAP begins there and the draws become its retries."""

    gradient_tolerance: Annotated[float, pydantic.Field(gt=0.0)] = 1e-5
    """Early stop on `|grad U(z)|_inf`, absolute in the unconstrained coordinates, which
    the transforms make O(1); the leaky-ReLU surrogate puts the minimum on a kink where
    the gradient is one side's slope, so there it is a diagnostic only."""

    n_starts: Annotated[int, pydantic.Field(ge=1)] = 1
    """MAP runs per inference, each started at its own draw from the priors; records
    stored before 2026-10-02 with one start began at the fixed `initial_values` instead.

    Runs get `DEFAULT_N_MAP_STARTS` through `PlasmaEstimationSettings`.
    """


class McmcSettings(pydantic.BaseModel):
    """NUTS settings, read only by `mcmc_nuts`.

    Every leapfrog step costs one gradient of the log-posterior, so the defaults are
    what a reference run can afford rather than what a sampler would like.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    n_warmup: Annotated[int, pydantic.Field(ge=1)] = 500
    """Warmup iterations, during which the step size and mass matrix adapt."""

    n_samples: Annotated[int, pydantic.Field(ge=1)] = 1000
    """Retained draws per chain."""

    n_chains: Annotated[int, pydantic.Field(ge=1)] = 2
    """Chains run.

    Two is the fewest that give an `r_hat`, which is what says the
    reference can be used as one; a single chain reports nan for it.
    """

    max_tree_depth: Annotated[int, pydantic.Field(ge=1)] = 6
    """Cap on the doubling depth, so one iteration costs at most `2^depth - 1`
    gradients. The cap bounds the run time and, when it binds often, biases the
    sampler, which is why the notebook reports how close the chains ran to it."""

    target_accept_prob: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)] = 0.8
    """Acceptance probability the step size adapts towards during warmup.

    NumPyro's own default. Raising it shortens the step and lengthens the trajectories,
    which is the standard response to divergences.
    """

    dense_mass: bool = False
    """Adapt a full mass matrix over the warmup instead of a diagonal one, so the steps
    follow strongly correlated parameters, such as the two `Te` exponents, along their
    ridge rather than across it; records stored before the field read as diagonal."""

    chain_init_jitter: Annotated[float, pydantic.Field(ge=0.0)] = 0.1
    """Standard deviation, in the unconstrained space, of the perturbation that
    separates the chains' start points around the MAP.

    Zero starts every chain at the same point, where `r_hat` no longer measures
    whether chains started elsewhere would have agreed. Read by `map` alone.
    """

    chain_init_width: Annotated[float, pydantic.Field(gt=0.0)] = 3.0
    """Laplace standard deviations the `laplace` starts lie from the MAP: each chain
    starts at a draw of the Laplace Gaussian widened by this factor, overdispersed
    relative to the posterior as `r_hat` assumes (author's decision, 2026-10-01)."""

    init_strategy: ChainInitStrategy = "laplace"
    """Where the chains start: `laplace` at independent draws of the widened Laplace
    Gaussian, `map` at the MAP perturbed by `chain_init_jitter` in `z`."""


class PlasmaEstimationSettings(pydantic.BaseModel):
    """Configuration of a single inference run.

    `profile_family` is the one modelling choice and the priors follow from it; whatever
    it is, `p = 2 ne Te` is resampled onto the surrogate's own fixed pressure grid.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    method: InferenceMethod = "map_laplace"
    """Where the band around the MAP comes from; see `InferenceMethod`.

    The default is the cheapest and the only deterministic recipe.
    """

    profile_family: profile_parametrization_types.ProfileFamily = "two_power_zero_edge"
    """Which parametrization the `ne`/`Te` profiles are inferred in; the only inferable
    one is `two_power_zero_edge`."""

    inverse_mapping_maxiter: Annotated[int, pydantic.Field(ge=1)] = 5
    """Fixed Newton steps in the real-space to flux-coordinate inverse map.

    Lower is faster but risks non-convergence, reported as a residual-above-tolerance
    warning. Five is enough only because the seed is precomputed and the boundary fixed.
    """

    interferometer_n_sample: Annotated[int, pydantic.Field(ge=2)] = 16
    """Points per interferometer chord in the line integral.

    Assumes `trim_chords_to_plasma`, so every sample lies inside the plasma and 16
    resolves the line integral to ~0.15%. Far too coarse without trimming.
    """

    trim_chords_to_plasma: bool = True
    """Replace each chord by its in-plasma segment before sampling.

    Valid because the surrogate is fixed-boundary, so the segment is computed once per
    run. False recovers the untrimmed behaviour; raise `interferometer_n_sample` too.
    """

    thomson_calibration_bound: Annotated[float, pydantic.Field(ge=1.0)] = (
        thomson_scattering_utils.CALIBRATION_BOUND
    )
    """`N` of the log-uniform prior on every correction, `delta_j` in `[1/N, N]`
    (`priors_utils.calibration_priors`), the interferometer constraining them.

    One takes the geometric calibration as known and infers no correction at all.
    """

    interferometer_sigma: Annotated[float, pydantic.Field(gt=0.0)] = (
        interferometer_utils.LINE_DENSITY_NOISE_M2
    )
    """Constant 1-sigma uncertainty `sigma_IF` of every interferometer chord in m^-2,
    independent of its reading; the default is Dreier's 0.1 rad phase uncertainty at
    10.6 um (`interferometer_utils.LINE_DENSITY_NOISE_M2`)."""

    n_posterior_samples: Annotated[int, pydantic.Field(ge=2)] = 512
    """Draws taken from the Laplace Gaussian at the reported MAP start to turn it into a
    profile band, and from each converged start's own Gaussian for its band width.

    Only the percentiles' Monte Carlo noise depends on it. A comparison against the
    sampler sets it to the sampler's own draw count, so the two bands are estimated from
    equally many draws.
    """

    map: MapSettings = MapSettings(n_starts=DEFAULT_N_MAP_STARTS)
    """Optimizer settings of the MAP every recipe starts from."""

    mcmc: McmcSettings = McmcSettings()
    """Sampler settings, used by `mcmc_nuts` only."""

    seed: int = 42
    """Seed of the synthetic noise draw and of every Monte Carlo draw of the run."""

    add_measurement_noise: bool = True
    """Whether the synthetic target carries a noise draw.

    False fits the expected signals with the noisy target's uncertainties: same band, no
    realization, so any offset left is representation or surrogate error.
    """

    verbose: bool = False
    """Print one line per MAP start."""


class PlasmaProfilePosterior(pydantic.BaseModel):
    """`ne`, `Te` and pressure on a shared `rho` grid: a mean, a median and a credible
    band, the mean being the reported profile.

    For a known truth the curves coincide and hold the true profile.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    rho: Annotated[jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer]
    """Radial grid all nine curves below are evaluated on."""

    electron_density_median: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Median electron density, in m^-3."""

    electron_density_lower: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Lower edge of the electron-density band."""

    electron_density_upper: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Upper edge of the electron-density band."""

    electron_temperature_median: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Median electron temperature, in eV."""

    electron_temperature_lower: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Lower edge of the electron-temperature band."""

    electron_temperature_upper: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Upper edge of the electron-temperature band."""

    pressure_median: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Median total thermal pressure, in Pa."""

    pressure_lower: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Lower edge of the pressure band."""

    pressure_upper: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Upper edge of the pressure band."""

    electron_density_mean: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Mean electron density, in m^-3."""

    electron_temperature_mean: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Mean electron temperature, in eV."""

    pressure_mean: Annotated[
        jt.Float[NpOrJaxArray, " n_rho"], array_validator, array_serializer
    ]
    """Mean total thermal pressure, in Pa."""


class PlasmaStateInferenceResult(pydantic.BaseModel):
    """Complete, self-contained record of one plasma-state inference run."""

    # NaN and inf written as JSON constants rather than null, so a run whose band
    # failed stays loadable.
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True, ser_json_inf_nan="constants"
    )

    study_name: str = ""
    """Folder name of the study that produced this run, e.g. `mcmc_benchmark`.

    Every study's runs share one store, so this is what a study's manifest selects on.
    Empty for a one-off run.
    """

    study_label: str = ""
    """Which cell of which study produced this run, e.g. `ts_10_if_05`.

    Empty for a one-off run. Set by a study so its manifest can be nothing but a map
    from label to id, and a stale manifest cannot disagree with the records.
    """

    dataset: str = ""
    """Stem of the truth's dataset, e.g. `equilibrium_a_3k_spline_input`."""

    thomson_volume_indices: list[int] | None = None
    """Which Thomson volumes the likelihood read, of those the instrument carries.

    None when every volume was read; this is what says which layout a run measured.
    `n_thomson_channels_masked` counts faint dropped channels, a different thing.
    """

    interferometer_chord_indices: list[int] | None = None
    """Which interferometer chords the likelihood read; None when all of them."""

    fixed_equilibrium_id: str = ""
    """Equilibrium whose geometry this run held frozen, skipping the surrogate.

    Empty for the self-consistent run, where the geometry follows the pressure at every
    parameter value; a non-empty id assumes an equilibrium instead of solving for one.
    """

    diagnostics_placement_id: str = ""
    """Content hash of the stored `DiagnosticPlacement` this run was measured with.

    The instrument itself, geometry, response with its band scales, and chords, lives
    under `results_plots_tables/DiagnosticPlacement/<id>/`, so a run can be rebuilt
    without recomputing the placement.
    """

    inference_id: str
    """Short random identifier of this run, and the name of its stored directory.

    A notebook names the id it wants at the top and loads that record, so a figure is
    traceable to one run rather than to whatever was computed last.
    """

    generated: str
    """When the run finished; recorded here so the id itself can stay short."""

    truth_equilibrium_id: str = ""
    """`desc_equilibrium_id` of the truth, for a run whose truth was named by id rather
    than by catalog entry; empty when the name is the catalog's and says it already."""

    truth_name: str
    """Ground truth the synthetic data came from, a key of `DESC_TRUTH_IDS`."""

    surrogate_name: str
    """Name of the stored surrogate used as the forward model."""

    machine_scale: surrogate_types.MachineScale
    """The machine the truth was read at and the surrogate evaluated at."""

    likelihood_quantities: list[str]
    """The compared observables, `thomson_signals` and/or `line_integrated_density`."""

    design_point: diagnostic_design_point.DiagnosticDesignPoint = pydantic.Field(
        default_factory=diagnostic_design_point.DiagnosticDesignPoint
    )
    """The instrument this run was fitted against: layout, calibration and noise.

    Stored because the noise model lives on the design point rather than in the
    settings, so without it a record does not say what uncertainty its band assumed.
    """

    settings: PlasmaEstimationSettings
    """The configuration this run used."""

    map_profiles: profile_parametrization_types.PlasmaProfiles
    """The MAP point estimate of the profile parameters, the same for every recipe."""

    inferred_profiles: PlasmaProfilePosterior
    """Inferred `ne`/`Te`/pressure median and credible band of the recipe's Gaussian."""

    initial_profiles: PlasmaProfilePosterior
    """The optimizer's starting profiles, as a zero-width band.

    Stored because a recovery figure is only readable next to what the fit started from,
    the start being deliberately below the truth.
    """

    true_profiles: PlasmaProfilePosterior
    """The ground-truth profiles on the same grid, as a zero-width band."""

    log_posterior_map: float = 0.0
    """`log p(z, data)` at the MAP in the unconstrained space, transform log-Jacobians
    included: the negated optimizer objective, of the start that reached the highest."""

    map_log_posterior_spread: float = 0.0
    """Highest minus lowest `log_posterior_map` over the MAP starts: zero when every
    start reached one mode, large when some stopped short."""

    n_map_starts_converged: int = 1
    """MAP starts that stopped before `map.max_iterations`, on a stalled fresh memory or
    on the gradient tolerance; zero marks a run whose point and band are the best capped
    start's, which every table and figure keeps."""

    map_converged: bool = True
    """Whether any start converged, i.e. `n_map_starts_converged > 0`."""

    n_map_modes: int = 0
    """Converged starts ending more than `inference_run_utils.DISTINCT_MODE_TOLERANCE`
    above the best in `U`: distinct local minima, reported rather than blended."""

    map_mode_gap: float = 0.0
    """Gap in `U` from the best converged start to the nearest of the `n_map_modes`;
    zero when there is none."""

    map_band_width_ratio_ne: float = float("nan")
    """Widest converged start's `ne` band width over the reported start's, a width being
    the integral over rho of the posterior standard deviation under that start's own
    Gaussian (`start_band_width_ne` in the arrays)."""

    map_band_width_ratio_te: float = float("nan")
    """The same ratio for `Te` (`start_band_width_te` in the arrays)."""

    map_gradient_norm: float = float("nan")
    """`|grad U(z)|_2` at the MAP."""

    map_gradient_inf_norm: float = float("nan")
    """`|grad U(z)|_inf` at the MAP, a diagnostic: on a kink of the surrogate it is one
    side's slope and does not vanish at the minimum."""

    map_newton_decrement: float = float("nan")
    """`g^T H^-1 g` at the best start's MAP, `H` the exact Hessian of `U`: the Laplace
    precision, part of the convergence test, inflated where `H` is near singular."""

    chi_squared: float = 0.0
    """`sum(residual^2)` over the unmasked channels at the MAP.

    Computed from `likelihood_utils.residuals`, at one extra forward evaluation per run.
    """

    n_degrees_of_freedom: int = 0
    """Unmasked channels minus fitted parameters.

    Zero or negative for a target with fewer channels than parameters, so a reader has
    to check this field before dividing `chi_squared` by it.
    """

    map_iterations: int = 0
    """L-BFGS iterations of the best start, `map.max_iterations` when the cap ended
    it."""

    information_gain_bits: float = float("nan")
    """Expected information gain of the data under the Laplace Gaussian, in bits
    (`laplace_approximation_utils.information_gain_bits`); nan for `mcmc_nuts`."""

    n_thomson_channels: int = 0
    """Thomson channels in the target, `n_volume * n_channel`."""

    n_thomson_channels_masked: int = 0
    """Channels the faint-channel mask dropped, carrying no likelihood weight."""

    map_time_seconds: float = 0.0
    """Wall-clock time of the MAP optimization."""

    laplace_time_seconds: float = 0.0
    """Wall-clock time of the Laplace fits of every start, which both recipes run."""

    mcmc_time_seconds: float = 0.0
    """Wall-clock time of warmup plus sampling; zero for every other recipe."""

    r_hat_max: float = float("nan")
    """Worst rank-normalized split `r_hat` over the sampled sites; nan for a single
    chain and for every recipe that does not sample."""

    ess_bulk_min: float = float("nan")
    """Smallest bulk effective sample size over the sampled sites; nan when nothing
    sampled."""

    ess_tail_min: float = float("nan")
    """Smallest tail effective sample size over the sampled sites, the count the 5th and
    95th percentiles of the reported band are resolved with."""

    n_divergences: int = 0
    """Divergent transitions of the sampler, zero for every other recipe.

    A run with any is a run whose geometry the sampler could not follow.
    """

    n_max_tree_depth: int = 0
    """Retained iterations whose trajectory saturated `mcmc.max_tree_depth` rather than
    ending on a U-turn."""

    max_tree_depth_fraction: float = float("nan")
    """`n_max_tree_depth` over the retained iterations; nan when nothing sampled."""

    accept_prob_mean: float = float("nan")
    """Mean acceptance probability over the retained draws, to be read against
    `mcmc.target_accept_prob`."""

    step_size_mean: float = float("nan")
    """Mean adapted leapfrog step size over the retained draws."""

    ebfmi_min: float = float("nan")
    """Smallest energy Bayesian fraction of missing information over the chains."""

    n_leapfrog_steps: int = 0
    """Gradient evaluations of the retained iterations, the hardware-independent cost of
    a sampled run."""

    nuts_converged: bool = False
    """Whether every criterion of `mcmc_convergence_utils` is met.

    False for a recipe that does not sample, which has no chains to converge; read it
    beside `settings.method`, and never read a sampled posterior without it.
    """

    total_time_seconds: float = 0.0
    """Wall-clock time of the whole run."""
