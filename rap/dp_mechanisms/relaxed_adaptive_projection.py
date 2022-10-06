"""RAP mechanism function."""

from math import log, sqrt

import jax.numpy as jnp
from jax import random

from rap.dp_mechanisms import adaptive_selection, gaussian, relaxed_projection
from rap.rap_inputs import RAPInputs
from rap.results_manager import ResultManager
from rap.utils import sparsemax


def evaluate(
    result_manager: ResultManager,
    rap_inputs: RAPInputs
) -> jnp.DeviceArray:
    """Evaluates the RAP mechanism.

    Args:
        result_manager (RAPResultManager): Manages the computation and storage of RAP
            evaluation results.
        rap_inputs (RAPInputs): All necessary input parameters to execute the RAP mechanism.

    Returns:
        jnp.DeviceArray: Final value of optimized synthetic dataset.
    """
    # Convert (epsilon, delta)-DP values to rho-zCDP value
    ldinv = log(1/rap_inputs.delta)
    rho = rap_inputs.epsilon + 2*(ldinv - sqrt(ldinv * (rap_inputs.epsilon + ldinv)))

    # Initialize uniform random synthetic dataset
    synthetic_dataset = sparsemax.sparsemax_on_dataset(
        rap_inputs.data_manager.generate_random_relaxed_dataset(
            rap_inputs.synthetic_dataset_size,
            rap_inputs.random_key
        ),
        rap_inputs.data_manager.onehot_transcoder.feature_onehot_index_range
    )
    random_key, _ = random.split(rap_inputs.random_key)

    if rap_inputs.T == 1 and rap_inputs.K == rap_inputs.workload_manager.num_queries:
        synthetic_dataset = _non_adaptive_optimize(
            result_manager,
            rap_inputs,
            synthetic_dataset,
            rho,
            random_key
        )
    else:
        synthetic_dataset = _adaptive_optimize(
            result_manager,
            rap_inputs,
            synthetic_dataset,
            rho,
            random_key
        )
    result_manager.finalize()

    return synthetic_dataset

def _adaptive_optimize(
    result_manager: ResultManager,
    rap_inputs: RAPInputs,
    synthetic_dataset: jnp.DeviceArray,
    rho: float,
    random_key: jnp.DeviceArray,
) -> jnp.DeviceArray:
    """Primary evaluation loop for the adaptive case of the RAP mechanism (i.e., T>1 or K==m).

    Args:
        result_manager (RAPResultManager): Manages the computation and storage of RAP
            evaluation results.
        rap_inputs (RAPInputs): All necessary input parameters to execute the RAP mechanism.
        synthetic_dataset (jnp.DeviceArray): Initial synthetic dataset to be optimized.
        rho (float): rho zCDP privacy parameter.
        random_key (jnp.DeviceArray): JAX Randomness key.

    Returns:
        jnp.DeviceArray: Final value of optimized synthetic dataset.
    """
    # Initialize empty threshold arrays for selected queries and their answers
    selected_query_threshold_indices = {
        threshold: jnp.array([], dtype=jnp.int32)
        for threshold in rap_inputs.non_private_threshold_answers.keys()
    }
    selected_query_threshold_answers = {
        threshold: jnp.array([], dtype=jnp.float32)
        for threshold in rap_inputs.non_private_threshold_answers.keys()
    }

    # Execute T rounds of query selection and optimization
    for t in range(1, rap_inputs.T+1):
        # Select top K highest-error queries
        result_manager.timer.start()
        top_threshold_indices, top_threshold_noisy_answers = \
            adaptive_selection.evaluate_on_thresholds(
                rap_inputs.data_manager.num_rows,
                rap_inputs.workload_manager,
                rap_inputs.non_private_threshold_answers,
                selected_query_threshold_indices,
                selected_query_threshold_answers,
                synthetic_dataset,
                rap_inputs.K,
                rho/rap_inputs.T,
                random_key
            )
        random_key, _ = random.split(random_key)
        result_manager.timer.stop()

        # Run RP mechanism on t*K queries
        synthetic_dataset = relaxed_projection.evaluate_query_subset(
            result_manager.attach_new_rp_result_manager(),
            synthetic_dataset,
            rap_inputs.workload_manager,
            top_threshold_indices,
            top_threshold_noisy_answers,
            rap_inputs.optimizer_hyperparams,
            random_key
        )
        random_key, _ = random.split(random_key)
        result_manager.log_current_rp_result()

        # Update already-selected queries
        selected_query_threshold_indices = top_threshold_indices
        selected_query_threshold_answers = top_threshold_noisy_answers

    return synthetic_dataset

def _non_adaptive_optimize(
    result_manager: ResultManager,
    rap_inputs: RAPInputs,
    synthetic_dataset: jnp.DeviceArray,
    rho: float,
    random_key: jnp.DeviceArray,
) -> jnp.DeviceArray:
    """Primary evaluation logic for the non-adaptive case of the RAP mechanism (i.e., T==K==1).

    Args:
        result_manager (RAPResultManager): Manages the computation and storage of RAP
            evaluation results.
        rap_inputs (RAPInputs): All necessary input parameters to execute the RAP mechanism.
        synthetic_dataset (jnp.DeviceArray): Initial synthetic dataset to be optimized.
        rho (float): rho zCDP privacy parameter.
        random_key (jnp.DeviceArray): JAX Randomness key.

    Returns:
        jnp.DeviceArray: Final value of optimized synthetic dataset.
    """
    # Apply GM to all non-private answers
    result_manager.timer.start()
    privatized_threshold_answers = gaussian.evaluate_on_thresholds(
        rap_inputs.data_manager.num_rows,
        rap_inputs.non_private_threshold_answers,
        rho,
        random_key
    )
    random_key, _ = random.split(random_key)
    result_manager.timer.stop()

    # Execute RP mechanism on all queries
    synthetic_dataset = relaxed_projection.evaluate_all_queries(
        result_manager.attach_new_rp_result_manager(),
        synthetic_dataset,
        rap_inputs.workload_manager,
        privatized_threshold_answers,
        rap_inputs.optimizer_hyperparams,
        random_key
    )
    random_key, _ = random.split(random_key)
    result_manager.log_current_rp_result()

    return synthetic_dataset
