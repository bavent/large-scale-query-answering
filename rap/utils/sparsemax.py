"""Performs Sparsemax transformation."""

from functools import partial
from typing import List, Tuple

import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(1,))
def sparsemax_on_dataset(
    synthetic_dataset: jnp.DeviceArray,
    feature_index_ranges: Tuple[Tuple[int]]
) -> jnp.DeviceArray:
    """Performs Sparsemax transformation on each row of the given synthetic dataset.

    Args:
        synthetic_dataset (jnp.DeviceArray): Synthetic dataset to perform sparsemax transform on.

    Returns:
        jnp.DeviceArray: Transformed synthetic dataset.
    """
    return jnp.hstack(
        sparsemax_on_feature(synthetic_dataset[:,start_idx:end_idx])
        for start_idx, end_idx in feature_index_ranges
    )

@jit
def sparsemax_on_feature(feature_rows: jnp.DeviceArray) -> jnp.DeviceArray:
    """Sparsemax evaluation function (http://proceedings.mlr.press/v48/martins16.pdf).

    Args:
        feature_rows (jnp.DeviceArray): An array of dataset rows corresponding to a particular
            (relaxed) onehot encoding of a categorical feature.

    Returns:
        jnp.DeviceArray: Transformed set of rows using the sparsemax evaluation function.
    """
    # Sort feature values in descending order
    z = jnp.sort(feature_rows, axis=1)[:,::-1]

    # Compute k(z)
    z_cumsum = jnp.cumsum(z, axis=1)
    k = jnp.arange(1, 1+z.shape[1])
    set_condition = 1 + k * z > z_cumsum
    k_z = z.shape[1] - jnp.argmax(set_condition[:,::-1], axis=1)

    # Compute tau(z)
    tau_numerator = z_cumsum[jnp.arange(0, z.shape[0]), k_z-1] - 1
    tau_z = (tau_numerator / k_z).reshape(-1, 1)

    # Compute and return p
    p = jnp.maximum(0, feature_rows - tau_z)
    return p
