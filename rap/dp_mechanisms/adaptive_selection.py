"""Adaptive Selection mechanism function."""

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as onp
from jax import jit, random

from rap.dp_mechanisms import gaussian
from rap.threshold import Threshold
from rap.utils import top_k
from rap.workload_manager import WorkloadManager


def evaluate_on_thresholds(
    sensitive_dataset_size: jnp.int32,
    workload_manager: WorkloadManager,
    non_private_threshold_answers: Dict[Threshold, jnp.DeviceArray],
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
        workload_manager (WorkloadManager): Manager for workload of surrogate queries
            and their corresponding answering functionaility.
        non_private_threshold_answers (Dict[Threshold, jnp.DeviceArray]): Non-private answers to the
            queries on the underlying sensitive dataset.
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
    selected_threshold_top_k = {}
    for threshold, true_answers in non_private_threshold_answers.items():
        # Retrieve the already-computed answers to the unselected queries on the sensitive data
        already_selected_indices = already_selected_query_threshold_indices[threshold]
        unselected_true_answers = jnp.delete(true_answers, already_selected_indices, axis=0)

        # Answer all unselected surrogate queries on the synthetic dataset
        unselected_surrogate_answers = workload_manager.answer_queries_of_threshold(
            threshold,
            synthetic_dataset,
            exclude_indices=already_selected_indices
        )

        # Compute the error between the true answers and the synthetic surrogate answers
        unselected_errors = jnp.abs(unselected_true_answers - unselected_surrogate_answers)

        # Privately select the top-k highest error queries
        selected_threshold_top_k[threshold] = report_noisy_top_k(
            sensitive_dataset_size,
            unselected_errors,
            select_k,
            privacy_rho/2,
            random_key
        )
        #selected_threshold_top_k[threshold][0].block_until_ready()
        random_key, _ = random.split(random_key)

    # Distill into a single set of top-k queries
    flattened_selected_top_k = aggregate_and_flatten_top_k(selected_threshold_top_k, select_k)

    # Get true answers to selected top-k queries
    selected_top_k_true_answers = jnp.array([
        non_private_threshold_answers[threshold][selected_index]
        for threshold, selected_index, _ in flattened_selected_top_k
    ])

    # Answer the k selected queries with the GM
    selected_top_k_noisy_answers = gaussian.evaluate(
        sensitive_dataset_size,
        selected_top_k_true_answers,
        privacy_rho/2,
        random_key
    )
    random_key, _ = random.split(random_key)

    # Add query indices and answers into their thresholds' already-selected queries
    return unflatten_and_append_results(
        flattened_selected_top_k,
        already_selected_query_threshold_indices,
        already_selected_query_threshold_noisy_answers,
        selected_top_k_noisy_answers
    )

def aggregate_and_flatten_top_k(
    threshold_top_k: Dict[Threshold, Tuple[jnp.DeviceArray, jnp.DeviceArray]],
    select_k: jnp.int32
) -> Tuple[Tuple[Threshold, jnp.DeviceArray, jnp.DeviceArray]]:
    """Aggregates the top-k from each threshold into an aggregated and flatten list of
    overall top-k.

    Args:
        threshold_top_k (Dict[Threshold, Tuple[jnp.DeviceArray, jnp.DeviceArray]]): Dictionary
            mapping thresholds to their corresponding top-k query's indices and values.
        select_k (jnp.int32): Concrete value of k.

    Returns:
        Tuple[Tuple[Threshold, jnp.DeviceArray, jnp.DeviceArray]]: K sized list of tuples, where
            each tuple contains a threshold and a corresponding query index and value.
    """
    onp_threshold_top_k = {
        threshold: (onp.array(top_k_indices), onp.array(top_k_vals))
        for threshold, (top_k_indices, top_k_vals) in threshold_top_k.items()
    }
    flattened_arrays = []
    for threshold, (top_k_indices, top_k_vals) in onp_threshold_top_k.items():
        for i, index in enumerate(top_k_indices):
            flattened_arrays.append((threshold, index, top_k_vals[i]))
    flattened_arrays.sort(key=lambda x:x[2], reverse=True)
    flattened_top_k = tuple(flattened_arrays[:select_k])
    return flattened_top_k

def unflatten_and_append_results(
    flattened_results: Tuple[Tuple[Threshold, jnp.DeviceArray, jnp.DeviceArray]],
    threshold_stored_indices: Dict[Threshold, jnp.DeviceArray],
    threshold_stored_answers: Dict[Threshold, jnp.DeviceArray],
    flattened_noisy_answers: jnp.DeviceArray
) -> Tuple[Dict[Threshold, jnp.DeviceArray], Dict[Threshold, jnp.DeviceArray]]:
    """Unflattens a list of privatized results, appending each result (index and value) to its
    corresponding Threshold's arrays.

    Args:
        flattened_results (Tuple[Tuple[Threshold, jnp.DeviceArray, jnp.DeviceArray]]): List of
            Thresholds with each's corresponding query's index and (non-private) noisy answer.
        threshold_stored_indices (Dict[Threshold, jnp.DeviceArray]): Indices of the previously
            evaluated top error queries.
        threshold_stored_answers (Dict[Threshold, jnp.DeviceArray]): Privatized answers of the
            previously evaluated top error queries.
        flattened_noisy_answers (jnp.DeviceArray): List of privatized answers to the queries
            corresponding to the flattened_results.

    Returns:
        Tuple[Dict[Threshold, jnp.DeviceArray], Dict[Threshold, jnp.DeviceArray]]: Updated dicts
            mapping each threshold to their corresponding already-answered queries'
            indices and answers.
    """
    thresholds = list(threshold_stored_indices.keys())
    # Convert stored dicts of jax data to dicts of python lists
    threshold_index_list = {}
    threshold_answer_list = {}
    for threshold in thresholds:
        threshold_index_list[threshold] = list(onp.array(threshold_stored_indices[threshold]))
        threshold_answer_list[threshold] = list(onp.array(threshold_stored_answers[threshold]))
    # Add results to lists
    for i, (threshold, index, _) in enumerate(flattened_results):
        threshold_index_list[threshold].append(index)
        threshold_answer_list[threshold].append(flattened_noisy_answers[i])
    # Convert dict of python lists back to dicts of jax arrays
    for threshold in thresholds:
        threshold_stored_indices[threshold] = jnp.array(threshold_index_list[threshold], jnp.int64)
        threshold_stored_answers[threshold] = jnp.array(threshold_answer_list[threshold], jnp.float32)
    return threshold_stored_indices, threshold_stored_answers

@partial(jit, static_argnums=(2,))
def report_noisy_top_k(
    dataset_size: jnp.int32,
    results: jnp.DeviceArray,
    select_k: jnp.int32,
    privacy_rho: jnp.float64,
    random_key: jnp.DeviceArray
) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
    """Executes the Report Noisy Top-k mechanism on a given set of query errors.

    Args:
        dataset_size (jnp.int32): Number of datapoints in the privacy sensitive dataset.
        results (jnp.DeviceArray): Non-private results.
        select_k (jnp.int32): Number of queries to select.
        privacy_rho (jnp.float64): zCDP parameter.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        Tuple[jnp.DeviceArray, jnp.DeviceArray]: Indices of the top k results (descending), and
            their corresponding noisy values from selection randomness.
    """
    # Compute noise parameters
    beta_scale = jnp.sqrt(select_k / (2 * privacy_rho)) / dataset_size

    num_results = len(results)
    noise = random.gumbel(random_key, (num_results,), dtype=jnp.float32) * beta_scale
    noisy_results = results + noise
    noisy_top_k_vals, noisy_top_k_indices = top_k.top_k(noisy_results, select_k)

    return noisy_top_k_indices, noisy_top_k_vals
