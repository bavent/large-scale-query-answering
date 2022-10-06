"""Algorithm to find top-k indicies & values of a JAX array."""

from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit, lax


@partial(jit, static_argnums=(1,))
def top_k(data: jnp.DeviceArray, k: jnp.int32) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    """Top-k implementation utilizing scan functionality. Alternative to lax.top_k, which
    doesn't seem to work on GPUs.

    Args:
        data (jnp.DeviceArray): Data to extract top-k values from.
        k (jnp.int32): Number of top elements to extract.

    Returns:
        Tuple[jnp.DeviceArray, jnp.DeviceArray]: Indices of the top k values, and the top k values.
    """

    def top_1(data):
        index = jnp.argmax(data)
        val = data[index]
        data = data.at[index].set(-jnp.inf)
        return data, val, index

    def scannable_top_1(carry, unused):
        data = carry
        data, val, index = top_1(data)
        return data, (val, index)

    data, (vals, indices) = lax.scan(scannable_top_1, data, (), min(k, len(data)))

    return vals, indices
