import functools
from collections.abc import Callable, Mapping
from typing import NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import numpyro.distributions.transforms
import numpyro.infer.util
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)


class Posterior(NamedTuple):
    """A bound NumPyro model written as functions of one unconstrained vector `z`.

    Unconstrained because a bounded site is unbounded in `z`, where an optimizer needs
    no constraint, a Gaussian fits, and the support is automatic on return.
    """

    potential: Callable[[jt.Float[NpOrJaxArray, " n_param"]], jt.Float[jax.Array, ""]]
    """`U(z) = -log p(z, data)`, transform log-Jacobians included: what the MAP
    minimizes and the Laplace approximation expands."""

    constrain: Callable[[jt.Float[NpOrJaxArray, " n_param"]], dict]
    """Maps `z` back to the physical value of each parameter.

    Applies the stored transform of each parameter, e.g. the sigmoid that maps the real
    line onto the bounds of a uniform prior, without evaluating the forward models.
    """

    z_init: jt.Float[jax.Array, " n_param"]
    """The start point, i.e. the caller's initial values pushed into `z`."""

    site_names: tuple[str, ...]
    """Parameter names in the order of the entries of `z`, which is alphabetical:
    `calibration_0, calibration_1, calibration_10, ..., calibration_9, ne_0, ne_1, ne_2,
    te_0, te_1, te_2`."""

    transforms: dict[str, numpyro.distributions.transforms.Transform]
    """The transform of each parameter, kept so that a batch of draws can be mapped
    between `z` and the physical values without evaluating the forward models."""


def build_posterior(
    model: Callable[[], None],
    initial_values: Mapping[str, jt.Float[NpOrJaxArray, ""]],
) -> Posterior:
    """Write a bound NumPyro model as functions of one unconstrained vector `z`.

    Args:
        model: A no-argument model, i.e. `inference_model_utils.plasma_model`
            with its configuration bound by `functools.partial`.
        initial_values: Per-site values in the constrained space, fixing both the start
            point and which sites exist.

    Returns:
        The density, the transforms back, the start point and the coordinate order.
    """
    initial_unconstrained = numpyro.infer.util.unconstrain_fn(
        model, (), {}, {name: jnp.asarray(v) for name, v in initial_values.items()}
    )
    z_init, unravel = jax.flatten_util.ravel_pytree(initial_unconstrained)
    transforms = numpyro.infer.util.get_transforms(model, (), {}, initial_unconstrained)
    return Posterior(
        potential=functools.partial(_potential, model=model, unravel=unravel),
        constrain=functools.partial(_constrain, transforms=transforms, unravel=unravel),
        z_init=z_init,
        site_names=_site_names(initial_unconstrained),
        transforms=transforms,
    )


def _site_names(
    unconstrained: Mapping[str, jt.Float[NpOrJaxArray, ""]],
) -> tuple[str, ...]:
    """Which parameter each entry of `z` holds.

    One one-hot probe per parameter is flattened and its nonzero entry read off, so the
    answer cannot drift if JAX changes how it orders dictionary keys.

    Args:
        unconstrained: Per-site unconstrained values, all scalars.

    Returns:
        The parameter names, in the order of the entries of `z`.
    """
    names: list[str | None] = [None] * len(unconstrained)
    for name in unconstrained:
        probe = {key: jnp.asarray(1.0 if key == name else 0.0) for key in unconstrained}
        flat, _ = jax.flatten_util.ravel_pytree(probe)
        names[int(np.argmax(np.asarray(flat)))] = name
    if any(name is None for name in names):
        raise ValueError(
            "could not determine the coordinate order of z; every site must be a "
            "scalar, and two sites cannot share a slot."
        )
    return tuple(name for name in names if name is not None)


def _potential(
    z: jt.Float[NpOrJaxArray, " n_param"],
    model: Callable[[], None],
    unravel: Callable[[jt.Float[jax.Array, " n_param"]], dict],
) -> jt.Float[jax.Array, ""]:
    """`U(z) = -log p(z, data)`, including the transform log-Jacobians."""
    # Two `asarray` for two reasons: `unravel` comes from `ravel_pytree` and takes a
    # jax array, and `potential_energy` returns `Array | float`.
    return jnp.asarray(
        numpyro.infer.util.potential_energy(model, (), {}, unravel(jnp.asarray(z)))
    )


def _constrain(
    z: jt.Float[NpOrJaxArray, " n_param"],
    transforms: dict[str, numpyro.distributions.transforms.Transform],
    unravel: Callable[[jt.Float[jax.Array, " n_param"]], dict],
) -> dict:
    """Apply the transform of each parameter, returning the physical values."""
    return numpyro.infer.util.transform_fn(transforms, unravel(jnp.asarray(z)))
