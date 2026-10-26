# The bound NumPyro model as a flat function of one unconstrained vector, which is what
# the MAP, the Laplace fit and the sampler want instead of NumPyro's site dict.

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


class FlatPosterior(NamedTuple):
    """A bound NumPyro model reduced to functions of one flat unconstrained vector.

    Unconstrained because a bounded site is unbounded in `z`, where an optimizer needs
    no constraint, a Gaussian fits, and the support is automatic on return.
    """

    potential: Callable[[jt.Float[NpOrJaxArray, " n_param"]], jt.Float[jax.Array, ""]]
    """`U(z) = -log p(z, data)`, transform log-Jacobians included: what the MAP
    minimizes and the Laplace approximation expands."""

    constrain: Callable[[jt.Float[NpOrJaxArray, " n_param"]], dict]
    """Flat unconstrained vector to the physical per-site values.

    Applies the per-site bijections only, rather than re-running the whole model as
    NumPyro's `constrain_fn` would, so it does not cost a flux solve per call.
    """

    z_init: jt.Float[jax.Array, " n_param"]
    """The start point, i.e. the caller's initial values pushed into `z`."""

    site_names: tuple[str, ...]
    """Site names in the flat vector's own coordinate order, which `ravel_pytree` sorts
    by key and not by site index, so `ne_10` comes before `ne_2`. Measured with one-hot
    probes, not assumed, since confusing the two permutes every per-site array."""

    transforms: dict[str, numpyro.distributions.transforms.Transform]
    """The per-site bijections, kept so a batch of draws can be moved between the two
    spaces without re-running the model, which would cost one flux solve per draw."""


def build_flat_posterior(
    model: Callable[[], None],
    initial_values: Mapping[str, jt.Float[NpOrJaxArray, ""]],
) -> FlatPosterior:
    """Flatten a bound NumPyro model into functions of one unconstrained vector.

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
    return FlatPosterior(
        potential=functools.partial(_potential, model=model, unravel=unravel),
        constrain=functools.partial(_constrain, transforms=transforms, unravel=unravel),
        z_init=z_init,
        site_names=_flat_site_names(initial_unconstrained),
        transforms=transforms,
    )


def unconstrain_samples(
    posterior: FlatPosterior,
    samples: Mapping[str, jt.Float[NpOrJaxArray, " n_sample"]],
) -> jt.Float[jax.Array, "n_sample n_param"]:
    """Push a batch of constrained per-site draws into flat unconstrained vectors.

    Applies the stored inverse bijections, pure elementwise maths, so a whole batch
    converts at no model cost; `unconstrain_fn` per draw would re-run the model.

    Args:
        posterior: The flattened posterior whose transforms and order are used.
        samples: Per-site draws, each of shape `(n_sample,)`.

    Returns:
        One row per draw, in `posterior.site_names` order.
    """
    return jnp.stack(
        [
            jnp.asarray(posterior.transforms[name].inv(jnp.asarray(samples[name])))
            for name in posterior.site_names
        ],
        axis=-1,
    )


def _flat_site_names(
    unconstrained: Mapping[str, jt.Float[NpOrJaxArray, ""]],
) -> tuple[str, ...]:
    """Which site each slot of the flat vector holds, measured rather than assumed.

    One one-hot probe per site is raveled and its nonzero slot read off, so the answer
    cannot drift if JAX changes how it orders dictionary keys.

    Args:
        unconstrained: Per-site unconstrained values, all scalars.

    Returns:
        The site names, in the flat vector's coordinate order.
    """
    names: list[str | None] = [None] * len(unconstrained)
    for name in unconstrained:
        probe = {key: jnp.asarray(1.0 if key == name else 0.0) for key in unconstrained}
        flat, _ = jax.flatten_util.ravel_pytree(probe)
        names[int(np.argmax(np.asarray(flat)))] = name
    if any(name is None for name in names):
        raise ValueError(
            "could not determine the flat coordinate order; every site must be a "
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
    """Apply the per-site bijections, returning the physical values."""
    return numpyro.infer.util.transform_fn(transforms, unravel(jnp.asarray(z)))
