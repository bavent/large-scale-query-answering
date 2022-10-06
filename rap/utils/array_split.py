"""Faster implementation of array_split (for our use-case) than JAX's."""

from typing import List

import jax.numpy as jnp


def fast_split(
    array: jnp.DeviceArray,
    num_sections: int
) -> List[jnp.DeviceArray]:
    """Splits an array into approximately even sized chunks. Mirror's JAX's array_split
    implementation, only removing the use of lax.slice (because that was degrading
    performance for some reason).

    Args:
        array (jnp.DeviceArray): Array to be split into sections.
        num_sections (int): Number of sections to split array into.

    Returns:
        List[jnp.DeviceArray]: List of sub-arrays of approximately equal size.
    """
    num_rows = len(array)
    part_size, r = divmod(num_rows, num_sections)
    if r == 0:
        split_indices = jnp.arange(num_sections+1, dtype=jnp.uint32) * part_size
    else:
        split_indices = jnp.concatenate([
            jnp.arange(r+1, dtype=jnp.uint32) * (part_size+1),
            jnp.arange(num_sections-r, dtype=jnp.uint32) * part_size + ((r+1) * (part_size+1) - 1)
        ])
    return [
        array[start:end]
        for start, end in zip(split_indices[:-1], split_indices[1:])
    ]
