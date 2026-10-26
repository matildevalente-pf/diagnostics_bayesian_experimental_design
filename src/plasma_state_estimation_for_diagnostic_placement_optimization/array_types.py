import jax
import jaxtyping as jt
import numpy as np
import pydantic

# Globally enable 64-bit mode in JAX whenever this module is imported: the physics
# forward models and their validation tolerances assume float64.
jax.config.update("jax_enable_x64", True)

NpOrJaxArray = np.ndarray | jax.Array
"""Either a numpy or a JAX array, for jaxtyping: `Float[NpOrJaxArray, "3"]`."""

ScalarFloat = float | jt.Float[NpOrJaxArray, " "]
"""Either a plain Python float or a zero-dimensional array."""


def _as_numpy(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


array_validator = pydantic.BeforeValidator(_as_numpy)
"""Coerces JSON lists back to numpy arrays on pydantic validation."""

array_serializer = pydantic.PlainSerializer(
    lambda value: np.asarray(value).tolist(), return_type=list, when_used="json"
)
"""Serializes array fields to JSON lists; annotate array fields with both helpers."""
