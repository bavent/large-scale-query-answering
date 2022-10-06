"""Adaptive Selection mechanism function."""

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import random
from rap.dp_mechanisms import gaussian, report_noisy_max
from rap.threshold import Threshold
from rap.workload_manager import WorkloadManager


def threshold_adaptive_selection(
    sensitive_dataset_size: jnp.int32,
    threshold_true_answers: Dict[Threshold, jnp.DeviceArray],
    workload_manager: WorkloadManager,
    already_selected_query_threshold_indices: Dict[Threshold, jnp.DeviceArray],
    already_selected_query_threshold_noisy_answers: Dict[Threshold, jnp.DeviceArray],
    synthetic_dataset: jnp.DeviceArray,
    select_k: jnp.int32,
    privacy_rho: jnp.float64,
    random_key: jnp.DeviceArray
) -> Tuple[Dict[Threshold, jnp.DeviceArray], Dict[Threshold, jnp.DeviceArray]]:
    """Adaptive Selection mechanism to select the top k highest-error queries.

    Args:
        sensitive_dataset_size (jnp.int32): Size of the underlying sensitive dataset whose
            true answers are being used.
        threshold_true_answers (Dict[Threshold, jnp.DeviceArray]): Non-private answers to the
            queries on the underlying sensitive dataset.
        workload_manager (WorkloadManager): Manager for workload of surrogate queries
            and their corresponding answering functionaility.
        already_selected_query_threshold_indices (Dict[Threshold, jnp.DeviceArray]): Set of
            surrogate queries that have already been selected.
        already_selected_query_threshold_noisy_answers (Dict[Threshold, jnp.DeviceArray]): Noisy
            answers to the already-selected queries.
        synthetic_dataset (jnp.DeviceArray): Synthetic dataset used to evaluate
            the surrogate queries.
        select_k (jnp.int32): Number of queries to select.
        privacy_rho (jnp.float64): zCDP parameter.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        Tuple[Dict[Threshold, jnp.DeviceArray], Dict[Threshold, jnp.DeviceArray]]: Dictionaries
            mapping thresholds to the aggregate top k highest-error query indices and answers.
    """
    # Initialize dict of empty arrays to store already-selected threshold indices
    k_selected_query_threshold_indices = {
        threshold: []
        for threshold in threshold_true_answers.keys()
    }

    # Select top k queries
    for j in range(select_k):
        # Determine unselected queries of each threshold
        unselected_query_threshold_indices = {}
        for threshold, true_answers in threshold_true_answers.items():
            # Initialize index set for all possible queries of current threshold
            all_indices = jnp.arange(len(true_answers))
            already_selected_indices = already_selected_query_threshold_indices[threshold]
            unselected_indices = jnp.setdiff1d(
                all_indices,
                already_selected_indices,
                assume_unique=True
            )
            unselected_query_threshold_indices[threshold] = unselected_indices

        # Retrieve the already-computed answers to the unselected queries on the sensitive data
        unselected_true_query_threshold_answers = {
            threshold: true_answers[unselected_query_threshold_indices[threshold]]
            for threshold, true_answers in threshold_true_answers.items()
        }

        # Answer all unselected surrogate queries on the synthetic dataset
        unselected_surrogate_threshold_answers = workload_manager.answer_surrogate_queries_of_all_thresholds(
            synthetic_dataset,
            exclude_indices=already_selected_query_threshold_indices
        )

        # Compute the error between the true answers and the synthetic surrogate answers
        unselected_threshold_errors = {
            threshold: jnp.abs(unselected_true_answers - unselected_surrogate_threshold_answers[threshold])
            for threshold, unselected_true_answers in unselected_true_query_threshold_answers.items()
        }

        # Privately select the query that has greatest error with RNM
        selected_threshold, selected_query_index, selected_query_noisy_val = report_noisy_max.evaluate_on_thresholds(
            sensitive_dataset_size,
            unselected_threshold_errors,
            privacy_rho/2,
            random_key
        )
        random_key, _ = random.split(random_key)

        # Answer the newly selected query with GM
        selected_query_true_answer = \
            threshold_true_answers[selected_threshold][selected_query_index].reshape(1)
        selected_query_noisy_answer = gaussian.evaluate(
            sensitive_dataset_size,
            selected_query_true_answer,
            privacy_rho/2,
            random_key
        )
        random_key, _ = random.split(random_key)

        # Add query index and answer already-selected queries
        k_selected_query_threshold_indices[selected_threshold].append(selected_query_index)
        already_selected_query_threshold_indices[selected_threshold] = jnp.append(
            already_selected_query_threshold_indices[selected_threshold],
            selected_query_index
        )
        already_selected_query_threshold_noisy_answers[selected_threshold] = jnp.append(
            already_selected_query_threshold_noisy_answers[selected_threshold],
            selected_query_noisy_answer
        )

    return already_selected_query_threshold_indices, already_selected_query_threshold_noisy_answers

def adaptive_selection(
    sensitive_dataset_size: jnp.int32,
    true_answers: jnp.DeviceArray,
    workload_manager: WorkloadManager,
    already_selected_query_indices: jnp.DeviceArray,
    already_selected_query_noisy_answers: jnp.DeviceArray,
    synthetic_dataset: jnp.DeviceArray,
    select_k: jnp.int32,
    privacy_rho: jnp.float64,
    random_key: jnp.DeviceArray
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    """Adaptive Selection mechanism to select the top k highest-error queries.

    Args:
        sensitive_dataset_size (jnp.int32): Size of the underlying sensitive dataset whose
            true answers are being used.
        true_answers (jnp.DeviceArray): Non-private answers to the queries on the underlying
            sensitive dataset.
        workload_manager (WorkloadManager): Manager for workload of surrogate queries
            and their corresponding answering functionaility.
        already_selected_query_indices (jnp.DeviceArray): Set of surrogate queries that have
            already been selected.
        already_selected_query_noisy_answers (jnp.DeviceArray): Answers to the already-selected
        synthetic_dataset (jnp.DeviceArray): Synthetic dataset used to evaluate the
            surrogate queries.
        select_k (jnp.int32): Number of queries to select.
        privacy_rho (jnp.float64): zCDP parameter.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        Tuple[jnp.DeviceArray, jnp.DeviceArray]: Top k highest-error query indices and answers.
    """
    # Initialize index set for all possible queries
    all_indices = jnp.arange(len(workload_manager.aggregated_onehot_queries))

    k_selected_query_indices = []
    for j in range(select_k):
        # Find the indices of the queries that haven't yet been selected
        unselected_query_indices = jnp.setdiff1d(
            all_indices,
            already_selected_query_indices,
            assume_unique=True
        )

        # Retrieve the already-computed answers to the unselected queries on the sensitive data
        unselected_true_answers = true_answers[unselected_query_indices]

        # Answer all unselected surrogate queries on the synthetic dataset
        unselected_surrogate_answers = workload_manager.answer_all_surrogate_queries(
            synthetic_dataset,
            exclude_indices=already_selected_query_indices
        )

        # Compute the error between the true answers and the synthetic surrogate answers
        unselected_errors = jnp.abs(unselected_true_answers - unselected_surrogate_answers)

        # Privately select the query that has greatest error with RNM
        selected_query_index, selected_query_noisy_val = report_noisy_max.evaluate(
            sensitive_dataset_size,
            unselected_errors,
            privacy_rho/2,
            random_key
        )
        random_key, _ = random.split(random_key)

        # Answer the newly selected query with GM
        selected_query_true_answer = true_answers[selected_query_index].reshape(1)
        selected_query_noisy_answer = gaussian.evaluate(
            sensitive_dataset_size,
            selected_query_true_answer,
            privacy_rho/2,
            random_key
        )
        random_key, _ = random.split(random_key)

        # Add query index and answer already-selected queries
        k_selected_query_indices.append(selected_query_index)
        already_selected_query_indices = jnp.append(
            already_selected_query_indices,
            selected_query_index
        )
        already_selected_query_noisy_answers = jnp.append(
            already_selected_query_noisy_answers,
            selected_query_noisy_answer
        )
        # jax.profiler.save_device_memory_profile(f"as-profiling/memory-nd-{j}.prof")

    return already_selected_query_indices, already_selected_query_noisy_answers
