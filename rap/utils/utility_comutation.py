from typing import Dict, Tuple

import jax.numpy as jnp

from rap.threshold import Threshold


def compute_present_error(
    true_threshold_answers: Dict[Threshold, jnp.DeviceArray],
    predicted_threshold_answers: Dict[Threshold, jnp.DeviceArray]
) -> float:
    """Computes the present error for answers to all consistent queries of each given Threshold.

    Args:
        true_threshold_answers (Dict[Threshold, jnp.DeviceArray]): Dict mapping Thresholds to
            their corresponding consistent queries' true answers.
        predicted_threshold_answers (Dict[Threshold, jnp.DeviceArray]): Dict mapping Thresholds to
            their corresponding consistent queries' predicted answers.

    Returns:
        float: Max absolute error over all consistent queries.
    """
    all_max_errors = []
    for threshold, true_answers in true_threshold_answers.items():
        predicted_answers = predicted_threshold_answers[threshold]
        all_max_errors.append(jnp.max(jnp.abs(true_answers - predicted_answers)))
    overall_max_error = jnp.max(jnp.array(all_max_errors))
    return overall_max_error

def compute_future_error(
    true_threshold_answers: Dict[Threshold, jnp.DeviceArray],
    predicted_threshold_answers: Dict[Threshold, jnp.DeviceArray]
) -> Tuple[float, float]:
    """Computes the future error for answers to all consistent queries of each given Threshold.

    Args:
        true_threshold_answers (Dict[Threshold, jnp.DeviceArray]): Dict mapping Thresholds to
            their corresponding consistent queries' true answers.
        predicted_threshold_answers (Dict[Threshold, jnp.DeviceArray]): Dict mapping Thresholds to
            their corresponding consistent queries' predicted answers.

    Returns:
        float: Average over each Threshold of the maximum error to that Threshold's
            consistent queries.
    """
    all_max_errors = []
    for threshold, true_answers in true_threshold_answers.items():
        predicted_answers = predicted_threshold_answers[threshold]
        all_max_errors.append(jnp.max(jnp.abs(true_answers - predicted_answers)))
    all_max_errors = jnp.array(all_max_errors)
    average_max_error = jnp.mean(all_max_errors)
    std_dev_max_error = jnp.std(all_max_errors, ddof=1)
    return average_max_error, std_dev_max_error
