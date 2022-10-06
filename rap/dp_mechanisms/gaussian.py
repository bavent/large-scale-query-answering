"""Gaussian mechanism functionality."""

from typing import Dict

import jax.numpy as jnp
from jax import jit, random

from rap.threshold import Threshold


def evaluate_on_thresholds(
    dataset_size: jnp.int32,
    threshold_results: Dict[Threshold, jnp.DeviceArray],
    privacy_rho: jnp.float64,
    random_key: jnp.DeviceArray
) -> Dict[Threshold, jnp.DeviceArray]:
    """Executes the Gaussian mechanism on a distributed set of query results.

    Args:
        dataset_size (jnp.int32): Number of datapoints in the privacy sensitive dataset.
        threshold_results (Dict[Threshold, jnp.DeviceArray]): Non-private results of each threshold.
        privacy_rho (jnp.float64): zCDP parameter.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        Dict[Threshold, jnp.DeviceArray]: Privatized query results of each threshold.
    """
    # Split random key across thresholds
    split_random_keys = random.split(random_key, len(threshold_results))

    # Compute noise parameters
    num_results_per_threshold = [len(result) for result in threshold_results.values()]
    total_num_results = sum(num_results_per_threshold)
    std_dev = (jnp.sqrt(total_num_results / (2 * privacy_rho)) / dataset_size).astype(jnp.float32)

    # Add Gaussian noise to each thresholds' results
    privatized_threshold_results = {}
    for i, (threshold, result) in enumerate(threshold_results.items()):
        random_key = split_random_keys[i]
        num_results = num_results_per_threshold[i]
        normal_noise = random.normal(random_key, (num_results,), dtype=jnp.float32) * std_dev
        privatized_threshold_results[threshold] = jnp.clip(result + normal_noise, a_min=0, a_max=1)

    return privatized_threshold_results

@jit
def evaluate(
    dataset_size: jnp.int32,
    query_results: jnp.DeviceArray,
    privacy_rho: jnp.float64,
    random_key: jnp.DeviceArray
) -> jnp.DeviceArray:
    """Executes the Gaussian mechanism on a given set of queries.

    Args:
        dataset_size (jnp.int32): Number of datapoints in the privacy sensitive dataset.
        query_results (jnp.DeviceArray): Non-private results.
        privacy_rho (jnp.float64): zCDP parameter.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        jnp.DeviceArray: Privatized query results.
    """
    num_results = len(query_results)
    std_dev = jnp.sqrt(num_results / (2 * privacy_rho)) / dataset_size
    normal_noise = random.normal(random_key, (num_results,), dtype=jnp.float32) * std_dev
    return jnp.clip(query_results + normal_noise, a_min=0, a_max=1)
