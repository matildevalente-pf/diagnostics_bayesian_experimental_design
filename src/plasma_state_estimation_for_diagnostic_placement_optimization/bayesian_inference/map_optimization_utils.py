# MAP estimation in the unconstrained space: L-BFGS with a zoom line search on
# `U(z) = -log p(z, data)`, transform log-Jacobians included.

import functools
import time
from collections.abc import Callable, Iterator
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
from plasma_state_estimation_for_diagnostic_placement_optimization.bayesian_inference import (  # noqa: E501
    inference_model_utils,
    inference_target_types,
    inference_types,
    posterior_utils,
    priors_utils,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.ideal_mhd import (
    desc_types,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    model_storage_utils,
)

LBFGS_MEMORY_SIZE = 50
"""Curvature pairs L-BFGS keeps, more than the parameter count, so none is dropped."""

MAX_LINESEARCH_STEPS = 20
"""Trial steps the zoom line search may take in one iteration before it gives up."""

Potential = Callable[[jt.Float[jax.Array, " n_param"]], jt.Float[jax.Array, ""]]
"""`z -> U(z)`, the negative log posterior in the unconstrained space."""

StopReason = Literal["gradient_tolerance", "stalled", "iteration_cap"]
"""What ended a start: `|grad U|_inf` below `MapSettings.gradient_tolerance`, a failed
line search or a step below `MapSettings.objective_tolerance` on a fresh memory (the
converged end on a surrogate with kinks), or `MapSettings.max_iterations`."""


class MapResult(NamedTuple):
    """What one MAP optimization produced."""

    posterior: posterior_utils.Posterior
    """The posterior the MAP was found on, with `z_init` at the MAP."""

    start_point: dict[str, jt.Float[jax.Array, ""]]
    """Where the optimization started, per site, in the constrained physical space."""

    point: dict[str, jt.Float[jax.Array, ""]]
    """The MAP estimate, per site, in the constrained physical space."""

    trajectory: dict[str, jt.Float[jax.Array, " n_iteration"]]
    """The estimate of each site after every L-BFGS iteration, constrained, for the
    convergence figure."""

    n_iterations: int
    """L-BFGS iterations run, memory resets included."""

    stop_reason: StopReason
    """What ended the start; see `StopReason`."""

    final_loss: float
    """`U(z)` at the MAP, i.e. the negative log posterior in the unconstrained space."""

    gradient_norm: float
    """`|grad U(z)|_2` at the end point."""

    gradient_inf_norm: float
    """`|grad U(z)|_inf` at the end point, the early-stop test and a diagnostic."""

    seconds: float
    """Wall-clock time of the optimization, compilation included."""


def run_map(
    settings: inference_types.PlasmaEstimationSettings,
    priors: priors_utils.ProfilePriors,
    target: inference_target_types.InferenceTarget,
    surrogate: model_storage_utils.LoadedSurrogate,
    starts: list[dict[str, jt.Float[jax.Array, ""]]],
    fixed_coefficients: desc_types.DescFourierZernikeCoefficients | None = None,
) -> Iterator[MapResult]:
    """Find the MAP from each start in turn, so a caller can stop at the first that
    converges: L-BFGS with a zoom line search (Nocedal and Wright 2006, section 7.2,
    Algorithm 7.5, and section 3.5, Algorithms 3.5-3.6), one compiled step for all
    starts.

    Args:
        settings: The run configuration, supplying the iteration cap and the two
            tolerances.
        priors: Priors on both profiles.
        target: The observed data.
        surrogate: The equilibrium forward model.
        starts: Where each run starts, one value per sample site in the constrained
            space, `inference_model_utils.start_points` and any warm start.
        fixed_coefficients: Geometry to hold frozen; None solves for it. See
            `inference_model_utils.plasma_model`.

    Yields:
        One result per start, in start order: its MAP point, its trajectory, what
        ended it and how stationary the point is.
    """
    posterior = posterior_utils.build_posterior(
        inference_model_utils.bind_plasma_model(
            settings, priors, target, surrogate, fixed_coefficients
        ),
        starts[0],
    )
    # `approx_dec_rtol=None` keeps the line search to the exact sufficient-decrease
    # condition: its approximate one accepts a step that raises `U` by up to 1e-6 of
    # its value, which the reset rule would read as a stall.
    optimizer = optax.lbfgs(
        memory_size=LBFGS_MEMORY_SIZE,
        scale_init_precond=False,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=MAX_LINESEARCH_STEPS,
            initial_guess_strategy="one",
            approx_dec_rtol=None,
        ),
    )
    # Bound once and jitted once, so the step compiles on the first start and every
    # later start reuses it.
    step = jax.jit(
        functools.partial(
            _lbfgs_step,
            optimizer=optimizer,
            value_and_grad=optax.value_and_grad_from_state(posterior.potential),
            potential=posterior.potential,
        )
    )
    constrain = jax.jit(posterior.constrain)
    potential = jax.jit(posterior.potential)
    gradient = jax.jit(jax.grad(posterior.potential))
    initial_points = jnp.stack(
        [
            jnp.asarray(
                posterior.transforms[name].inv(
                    jnp.stack([start[name] for start in starts])
                )
            )
            for name in posterior.site_names
        ],
        axis=-1,
    )
    for index, (start, z) in enumerate(zip(starts, initial_points)):
        started = time.perf_counter()
        z, history, stop_reason = _lbfgs(z, settings, optimizer, step, constrain)
        final_gradient = gradient(z)
        gradient_norm = float(jnp.linalg.norm(final_gradient))
        gradient_inf_norm = float(jnp.max(jnp.abs(final_gradient)))
        final_loss = float(potential(z))
        if settings.verbose:
            print(
                f"[MAP] start {index}: U(z) = {final_loss:.6e} after {len(history)} "
                f"L-BFGS iterations ({stop_reason}), |grad U|_inf = "
                f"{gradient_inf_norm:.2e}"
            )
        point = constrain(z)
        yield MapResult(
            posterior=posterior._replace(z_init=z),
            start_point=start,
            point=point,
            trajectory={
                site: jnp.stack([entry[site] for entry in history]) for site in point
            },
            n_iterations=len(history),
            stop_reason=stop_reason,
            final_loss=final_loss,
            gradient_norm=gradient_norm,
            gradient_inf_norm=gradient_inf_norm,
            seconds=time.perf_counter() - started,
        )


def _lbfgs(
    z: jt.Float[jax.Array, " n_param"],
    settings: inference_types.PlasmaEstimationSettings,
    optimizer: optax.GradientTransformationExtraArgs,
    step: Callable,
    constrain: Callable,
) -> tuple[jt.Float[jax.Array, " n_param"], list[dict], StopReason]:
    """L-BFGS from `z`: a failed line search or a step below the objective tolerance
    resets the memory, and the same on a fresh memory ends the start converged, as does
    `|grad U|_inf` below the gradient tolerance; the cap ends it unconverged."""
    state = optimizer.init(z)
    fresh = True
    history: list[dict] = []
    for _ in range(settings.map.max_iterations):
        z_new, state_new, decrease, failed, gradient_inf_norm = step(z, state)
        if not failed:
            z, state = z_new, state_new
        history.append(constrain(z))
        if not failed and float(gradient_inf_norm) < settings.map.gradient_tolerance:
            return z, history, "gradient_tolerance"
        if failed or float(decrease) < settings.map.objective_tolerance:
            if fresh:
                return z, history, "stalled"
            # A stale memory is the usual cause of a poor direction, so steepest
            # descent from the same point decides whether the start is done.
            state, fresh = optimizer.init(z), True
        else:
            fresh = False
    return z, history, "iteration_cap"


def _lbfgs_step(
    z: jt.Float[jax.Array, " n_param"],
    state: optax.OptState,
    optimizer: optax.GradientTransformationExtraArgs,
    value_and_grad: Callable,
    potential: Potential,
) -> tuple[
    jax.Array,
    optax.OptState,
    jt.Float[jax.Array, ""],
    jt.Bool[jax.Array, ""],
    jt.Float[jax.Array, ""],
]:
    """One L-BFGS iteration on `U(z)`: the new point, the state, how much the step
    lowered `U`, whether the line search found no step of sufficient decrease, and
    `|grad U|_inf` at the new point."""
    value, gradient = value_and_grad(z, state=state)
    updates, state = optimizer.update(
        gradient, state, z, value=value, grad=gradient, value_fn=potential
    )
    # The line search leaves `U` and its gradient at the point it accepted in the
    # state, where `value_and_grad` reads them back on the next iteration instead of
    # recomputing them, so the gradient test costs nothing.
    decrease = value - optax.tree_utils.tree_get(state, "value")
    failed = optax.tree_utils.tree_get(state, "decrease_error") > 0.0
    gradient_inf_norm = jnp.max(jnp.abs(optax.tree_utils.tree_get(state, "grad")))
    return (
        cast(jax.Array, optax.apply_updates(z, updates)),
        state,
        decrease,
        failed,
        gradient_inf_norm,
    )
