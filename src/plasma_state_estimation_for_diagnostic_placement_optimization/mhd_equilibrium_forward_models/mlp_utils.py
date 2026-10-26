from collections.abc import Sequence
from typing import cast

import flax.linen as nn
import flax.typing
import jax
import jax.numpy as jnp
from jaxtyping import Float
from plasma_state_estimation_for_diagnostic_placement_optimization.array_types import (
    NpOrJaxArray,
)
from plasma_state_estimation_for_diagnostic_placement_optimization.mhd_equilibrium_forward_models import (  # noqa: E501
    mlp_types,
    preprocessing_types,
    preprocessing_utils,
)


class MLP(nn.Module):
    """A fully connected network, from the normalized pressure to the network's latent.

    The module is passed to `jax.jit` as a static argument, so every field has to be
    hashable: build it with `build_mlp`, which turns the configured widths into a tuple.
    """

    hidden_layer_widths: Sequence[int]
    """Width of each hidden layer, in order."""

    n_outputs: int
    """Width of the output layer, i.e. the latent width of the fitted preprocessor."""

    dropout_rate: float
    """Dropout probability after each hidden layer; zero leaves the layers out."""

    activation: mlp_types.ActivationName
    """Activation applied after each hidden layer."""

    @nn.compact
    def __call__(
        self,
        x: Float[NpOrJaxArray, "... n_rho"],
        training: bool = False,
    ) -> Float[jax.Array, "... n_latent"]:
        """Map normalized pressure profiles to the network's latent space.

        Args:
            x: One normalized pressure profile, or a batch of them.
            training: Whether dropout is active; it needs a `dropout` key in `rngs`.

        Returns:
            The latent vector of each input profile.
        """
        activation = getattr(nn, self.activation)
        for width in self.hidden_layer_widths:
            x = nn.Dense(width)(x)
            x = activation(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        return nn.Dense(self.n_outputs)(x)


def build_mlp(architecture: mlp_types.MLPArchitecture, n_outputs: int) -> MLP:
    """Build the network one architecture describes.

    The configured widths are a list, which is unhashable, and the module is a static
    argument of the jitted training step, so they become a tuple here.

    Args:
        architecture: Widths, activation and dropout rate of the network.
        n_outputs: Width of the output layer.

    Returns:
        The network, ready for `init_params`.
    """
    return MLP(
        hidden_layer_widths=tuple(architecture.hidden_layer_widths),
        n_outputs=n_outputs,
        dropout_rate=architecture.dropout_rate,
        activation=architecture.activation,
    )


def init_params(mlp: MLP, n_inputs: int, key: jax.Array) -> dict:
    """Initialize the network's parameters.

    Args:
        mlp: Network to initialize.
        n_inputs: Number of pressure sample points the network takes.
        key: PRNG key of the initialization.

    Returns:
        The Flax parameter tree.
    """
    # float32 explicitly: this folder enables JAX float64 at import, so a default-dtype
    # dummy would initialize a float64 network and double the cost of every step.
    return dict(mlp.init(key, jnp.zeros((1, n_inputs), dtype=jnp.float32)))


def count_parameters(params: flax.typing.VariableDict) -> int:
    """Count the scalar weights in a parameter tree.

    Args:
        params: The Flax parameter tree.

    Returns:
        The total number of scalars.
    """
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def predict_y(
    mlp: MLP,
    params: flax.typing.VariableDict,
    preprocessor: preprocessing_types.PreprocessorParams,
    pressure: Float[NpOrJaxArray, "... n_rho"],
) -> Float[jax.Array, "... n_y"]:
    """Predict the flat target vector of one pressure profile, or of a batch of them.

    Traceable and differentiable in `pressure`. Bind everything but `pressure` with
    `functools.partial` to jit it, and unpack the result with
    `surrogate_utils.unpack_flat_target`.

    Args:
        mlp: Network to run, with dropout inactive.
        params: Its trained parameters.
        preprocessor: The normalization the network was trained under.
        pressure: Pressure on the basis' `pressure_rho_grid` [Pa].

    Returns:
        The flat target vector, laid out as `surrogate_types.FLAT_TARGET_LAYOUT`.
    """
    latent = cast(
        jax.Array,
        mlp.apply(
            params,
            preprocessing_utils.transform_x(preprocessor, pressure),
            training=False,
        ),
    )
    return jnp.asarray(preprocessing_utils.inverse_transform_y(preprocessor, latent))
