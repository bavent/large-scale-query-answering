"""Report Noisy Max mechanism function."""

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import random
from rap.threshold import Threshold


def evaluate_on_thresholds(
    dataset_size: jnp.int32,
    threshold_results: Dict[Threshold, jnp.DeviceArray],
    privacy_rho: jnp.float64,
    random_key: jnp.DeviceArray
) -> Tuple[Threshold, jnp.DeviceArray, jnp.DeviceArray]:
    """Executes the Report Noisy Max mechanism on a given set of query errors.

    Args:
        dataset_size (jnp.int32): Number of datapoints in the privacy sensitive dataset.
        threshold_results (Dict[Threshold, jnp.DeviceArray]): Non-private results of each threshold.
        privacy_rho (jnp.float64): zCDP parameter.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        Tuple[Threshold, jnp.DeviceArray, jnp.DeviceArray]: Threshold that the
            max error query belongs to, index of the query within the threshold, and noisy max
            value from selection randomness (not private, only used for debugging purposes).
    """
    # Split random key across thresholds
    split_random_keys = random.split(random_key, len(threshold_results))

    # Compute noise parameters
    beta_scale = 1 / (jnp.sqrt(2 * privacy_rho) * dataset_size)

    # Find noisy argmax on each threshold
    threshold_maxs = {}
    for i, (threshold, result) in enumerate(threshold_results.items()):
        random_key = split_random_keys[i]
        num_results = len(result)
        noise = random.gumbel(random_key, (num_results,), dtype=jnp.float32) * beta_scale
        noisy_result = result + noise
        max_index = jnp.argmax(noisy_result)
        noisy_max_val = noisy_result[max_index]
        threshold_maxs[threshold] = (max_index, noisy_max_val)

    # Aggregate results and determine the single max
    collected_maxs = [
        (threshold, max_index, noisy_max_val)
        for threshold, (max_index, noisy_max_val) in threshold_maxs.items()
    ]
    max_threshold, threshold_max_index, noisy_max_val = max(collected_maxs, key=lambda item:item[2])

    return max_threshold, threshold_max_index, noisy_max_val

def evaluate(
    dataset_size: jnp.int32,
    results: jnp.DeviceArray,
    privacy_rho: jnp.float64,
    random_key: jnp.DeviceArray
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    """Executes the Report Noisy Max mechanism on a given set of query errors.

    Args:
        dataset_size (jnp.int32): Number of datapoints in the privacy sensitive dataset.
        results (jnp.DeviceArray): Non-private results.
        privacy_rho (jnp.float64): zCDP parameter.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        Tuple[jnp.DeviceArray, jnp.DeviceArray]: Index of the max error query and noisy max value
            from selection randomness (not private, only used for debugging purposes).
    """
    beta_scale = 1 / (jnp.sqrt(2 * privacy_rho) * dataset_size)
    num_results = len(results)
    gumbel_noise = random.gumbel(random_key, (num_results,), dtype=jnp.float32) * beta_scale
    noisy_results = results + gumbel_noise
    max_index = jnp.argmax(noisy_results)
    return max_index, noisy_results[max_index]
