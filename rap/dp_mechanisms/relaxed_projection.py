"""RP mechanism function."""

from math import ceil, sqrt
from typing import Any, Callable, Dict, Tuple

from jax import jit, random, value_and_grad
import jax.numpy as jnp
import numpy as onp
import optax

from rap.optimization_hyperparameters import OptimizationHyperparameters
from rap.results_manager import RPResultManager
from rap.threshold import Threshold
from rap.utils import sparsemax
from rap.utils import array_split
from rap.workload_manager import WorkloadManager


def _core_evaluate(
    result_manager: RPResultManager,
    synthetic_dataset: jnp.DeviceArray,
    workload_manager: WorkloadManager,
    shuffled_queries_or_ids: jnp.DeviceArray,
    shuffled_answers: jnp.DeviceArray,
    get_query_batch: Callable[[jnp.DeviceArray], jnp.DeviceArray],
    compute_epoch_loss: Callable[[jnp.DeviceArray], float],
    optimizer_hyperparams: OptimizationHyperparameters
) -> jnp.DeviceArray:
    """Core evaluation method of the RP mechanism, primarily implementing the main gradient-based
    optimization loop. Unifies both some-queries and all-queries cases for RP.

    Args:
        result_manager (RPResultManager): Manages computation and storage of RP evaluation results.
        synthetic_dataset (jnp.DeviceArray): Initial value of dataset to be optimized.
        workload_manager (WorkloadManager): WorkloadManager which contains the functions to
            convert queries from their implicit representation as well as to answer queries.
        shuffled_queries_or_ids (jnp.DeviceArray): An array containing either the explicit set of
            of queries (for the some-queries RP case) or containing the implicit set of queries
            represented by their global IDs (for the all-queries RP case). The implicit_queries
            argument should be set accordingly.
        shuffled_answers (jnp.DeviceArray): An array containing the answers to the corresponding
             to-be-optimized queries.
        get_query_batch (Callable[[jnp.DeviceArray], jnp.DeviceArray]): Function to extract a batch
            of queries from a list of queries or query IDs.
        compute_epoch_loss (Callable[[jnp.DeviceArray], float]): Function to compute the loss for
            the given query set.
        optimizer_hyperparams (OptimizationHyperparameters): Hyperparameters for the
            gradient-based optimizer.

    Returns:
        jnp.DeviceArray: Final value of optimized synthetic dataset.
    """
    # Divide queries/IDs and answers into approximately equally-sized batches
    num_queries = len(shuffled_queries_or_ids)
    batch_size = min(num_queries, optimizer_hyperparams.max_batch_size)
    num_batches = ceil(num_queries / optimizer_hyperparams.max_batch_size)
    query_or_id_batches = array_split.fast_split(shuffled_queries_or_ids, num_batches)
    answer_batches = array_split.fast_split(shuffled_answers, num_batches)

    # Generate update function for gradient optimizer, scaling learning rate to batch size
    learning_rate = optimizer_hyperparams.base_learning_rate * sqrt(batch_size)
    update_fn, opt_state = _get_update_fn(
        synthetic_dataset,
        learning_rate,
        workload_manager
    )

    # Repeatedly run gradient-based optimizer on batches
    prior_epoch_loss = current_epoch_loss = jnp.inf
    for epoch_num in range(optimizer_hyperparams.max_epochs):
        for batch_num in range(num_batches):
            # Get batch of queries and answers
            query_or_id_batch = query_or_id_batches[batch_num]
            queries = get_query_batch(query_or_id_batch)
            answers = answer_batches[batch_num]

            # Apply optimizer update and Sparsemax projection to synthetic dataset
            synthetic_dataset, opt_state, batch_loss = update_fn(
                queries,
                synthetic_dataset,
                answers,
                opt_state
            )

            # Log result every once in a while
            if batch_num % result_manager.batches_per_eval == 0:
                result_manager.log_batch_statistics(batch_loss, epoch_num, batch_num)

        if epoch_num % result_manager.epochs_per_eval == 0:
            # Compute statistics and log results
            current_epoch_loss = compute_epoch_loss(synthetic_dataset)
            result_manager.log_epoch_statistics(synthetic_dataset, current_epoch_loss, epoch_num)
            # Stop early if training hasn't progressed much from previous epoch
            if prior_epoch_loss - current_epoch_loss < optimizer_hyperparams.convergence_tol:
                break

        prior_epoch_loss = current_epoch_loss

    return synthetic_dataset

def evaluate_all_queries(
    result_manager: RPResultManager,
    synthetic_dataset: jnp.DeviceArray,
    workload_manager: WorkloadManager,
    threshold_query_answers: Dict[Threshold, jnp.DeviceArray],
    optimizer_hyperparams: OptimizationHyperparameters,
    random_key: jnp.DeviceArray,
) -> jnp.DeviceArray:
    """Executes the RP mechanism on all queries, given the true answers to the queries.

    Args:
        result_manager (RPResultManager): Manages computation and storage of RP evaluation results.
        synthetic_dataset (jnp.DeviceArray): Initial value of dataset to be optimized.
        workload_manager (WorkloadManager): WorkloadManager which contains the functions to
            convert queries from their implicit representation as well as to answer queries.
        threshold_query_answers (Dict[Threshold, jnp.DeviceArray]): Answers to all the queries
            of each Threshold.
        optimizer_hyperparams (OptimizationHyperparameters): Hyperparameters for the
            gradient-based optimizer.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        jnp.DeviceArray: Final value of optimized synthetic dataset.
    """
    # Flatten answers
    flattened_answers = jnp.concatenate([
        answers
        for answers in threshold_query_answers.values()
    ])

    # Initialize shuffled indicies with numpy (more memory-efficient than JAX's) & shuffle answers
    onp_rng = onp.random.default_rng(int(random_key[0]))
    random_key, _ = random.split(random_key)
    shuffled_numpy_indices = onp_rng.permutation(
        onp.arange(len(flattened_answers), dtype=onp.uint32)
    )
    shuffled_indices = jnp.array(shuffled_numpy_indices, dtype=jnp.uint32)
    shuffled_answers = flattened_answers[shuffled_indices]
    del shuffled_numpy_indices

    # Define batch getter function for implicitly-represented queries
    get_query_batch = lambda batch: workload_manager.convert_global_ids_to_queries(batch)

    # Define epoch loss computation function for implicitly-represented queries
    def compute_epoch_loss(
        synthetic_dataset: jnp.DeviceArray,
    ) -> float:
        total_vals = 0
        unnormalized_loss = 0.
        for threshold, targets in threshold_query_answers.items():
            num_targets = len(targets)
            total_vals += num_targets
            predictions = workload_manager.answer_queries_of_threshold(
                threshold,
                synthetic_dataset
            )
            unnormalized_loss += (_mse_loss(predictions, targets) * num_targets)
        return unnormalized_loss / total_vals

    return _core_evaluate(
        result_manager,
        synthetic_dataset,
        workload_manager,
        shuffled_indices,
        shuffled_answers,
        get_query_batch,
        compute_epoch_loss,
        optimizer_hyperparams
    )

def evaluate_query_subset(
    result_manager: RPResultManager,
    synthetic_dataset: jnp.DeviceArray,
    workload_manager: WorkloadManager,
    threshold_query_indices: Dict[Threshold, jnp.DeviceArray],
    threshold_query_answers: Dict[Threshold, jnp.DeviceArray],
    optimizer_hyperparams: OptimizationHyperparameters,
    random_key: jnp.DeviceArray,
) -> jnp.DeviceArray:
    """Executes the RP mechanism on a specific subset of queries, given those queries
    and their true answers.

    Args:
        result_manager (RPResultManager): Manages computation and storage of RP evaluation results.
        synthetic_dataset (jnp.DeviceArray): Initial value of dataset to be optimized.
        workload_manager (WorkloadManager): WorkloadManager which contains the functions to
            convert queries from their implicit representation as well as to answer queries.
        threshold_query_indices (Dict[Threshold, jnp.DeviceArray]): For each Threshold,
            an array of the local IDs of the to-be-answered queries.
        threshold_query_answers (Dict[Threshold, jnp.DeviceArray]): For each Threshold,
            the true answers to the queries specified in threshold_query_indices.
        optimizer_hyperparams (OptimizationHyperparameters): Hyperparameters for the
            gradient-based optimizer.
        random_key (jnp.DeviceArray): JAX randomness key.

    Returns:
        jnp.DeviceArray: Final value of optimized synthetic dataset.
    """
    # Flatten query indicies and answers
    flattened_queries = jnp.concatenate([
        threshold.local_ids_to_queries(local_indices)
        for threshold, local_indices in threshold_query_indices.items()
    ])
    flattened_answers = jnp.concatenate([
        answers
        for answers in threshold_query_answers.values()
    ])

    # Shuffle queries and answers
    shuffled_indices = random.permutation(random_key, len(flattened_answers))
    random_key, _ = random.split(random_key)
    shuffled_queries = flattened_queries[shuffled_indices]
    shuffled_answers = flattened_answers[shuffled_indices]

    # Define batch getter function for implicitly-represented queries
    get_query_batch = lambda batch: batch

    # Define epoch loss computation function for implicitly-represented queries
    @jit
    def compute_epoch_loss(
        synthetic_dataset: jnp.DeviceArray,
    ) -> float:
        targets = flattened_answers
        predictions = workload_manager.answer_surrogate_queries(
            flattened_queries,
            synthetic_dataset
        )
        return _mse_loss(predictions, targets)

    return _core_evaluate(
        result_manager,
        synthetic_dataset,
        workload_manager,
        shuffled_queries,
        shuffled_answers,
        get_query_batch,
        compute_epoch_loss,
        optimizer_hyperparams
    )

def _get_update_fn(
    initial_state: jnp.DeviceArray,
    learning_rate: float,
    workload_manager: WorkloadManager
) -> Tuple[Callable, Any]:
    """Defines the function used to update the synthetic dataset via gradient optimization.

    Args:
        initial_state (jnp.DeviceArray): Initial synthetic dataset state.
        learning_rate (float): Learning rate for the gradient optimizer.
        workload_manager (WorkloadManager): WorkloadManager to provide query answering function
            and feature transcoding information.

    Returns:
        Tuple[Callable, Any]: Update function and its optimizer's initial state.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(initial_state)

    loss_fn = lambda queries, synthetic_dataset, answers: _mse_loss(
        workload_manager.answer_surrogate_queries(queries, synthetic_dataset),
        answers
    )

    # Define optimizer's update step
    @jit
    def update_fn(queries, synthetic_dataset, answers, opt_state):
        # Compute gradients of the loss wrt the synthetic dataset
        loss, grads = value_and_grad(loss_fn, 1)(
            queries,
            synthetic_dataset,
            answers
        )
        # Update the synthetic dataset based on the gradients, and updatethe optimizer state
        updates, opt_state = optimizer.update(grads, opt_state)
        synthetic_dataset = optax.apply_updates(synthetic_dataset, updates)
        # Apply SparseMax transform to updated synthetic dataset
        synthetic_dataset = sparsemax.sparsemax_on_dataset(
            synthetic_dataset,
            workload_manager.onehot_transcoder.feature_onehot_index_range
        )
        return synthetic_dataset, opt_state, loss

    return update_fn, opt_state

@jit
def _mse_loss(
    predictions: jnp.DeviceArray,
    targets: jnp.DeviceArray
) -> jnp.DeviceArray:
    """MSE loss function.

    Args:
        predictions (jnp.DeviceArray): Predicted values.
        targets (jnp.DeviceArray): "True" values being optimized towards.

    Returns:
        jnp.DeviceArray: Scalar loss value.
    """
    return jnp.sum((predictions - targets)**2) / len(predictions)
