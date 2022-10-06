"""Performs all high level management of a workload of r-of-k thresholds."""

import itertools
import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jit, lax, random, vmap
from scipy.special import binom
import scipy.stats as stats

from data.onehot_transcoder import OnehotTranscoder
import rap.utils.gpu_memory
from rap.threshold import Threshold


class WorkloadManager:
    """Manages a workload of r-of-k thresholds, including the ability to generate
    and answer them."""

    def __init__(
        self,
        r: int,
        k: int,
        num_thresholds:int,
        onehot_transcoder: OnehotTranscoder,
        threshold_size_limit: Optional[int] = None
    ) -> None:
        """Performs the necessary functions to initialize a random workload of r-of-k thresholds.

        Args:
            r (int): Parameter specifying the number of matching categories needed to surpass the
                threshold for any query.
            k (int): Parameter specifying the number of categorical features in any threshold.
            num_thresholds (int): The total number of thresholds in the workload.
            onehot_transcoder (OnehotTranscoder): Transcoder used to move between categorical and
                onehot representations of the data space.
            threshold_size_limit (Optional[int]): Maximum number of consistent queries that any
                selected threshold may contain.
        """
        # Threshold properties
        self.r = r
        self.k = k
        self.num_thresholds = num_thresholds
        self.threshold_size_limit = threshold_size_limit

        # Data properties
        self.onehot_transcoder = onehot_transcoder

        # Query evaluators
        self.categorical_predicate_evaluator = \
            Threshold.generate_categorical_predicate_evaluator(r, k)
        self.surrogate_predicate_evaluator = \
            Threshold.generate_surrogate_predicate_evaluator(r, k)

        # System properties
        self.total_device_memory = rap.utils.gpu_memory.total()

    def generate_future_workload(
        self,
        num_thresholds: int,
        random_key: jnp.DeviceArray
    ) -> "WorkloadManager":
        """Creates a new WorkloadManager object based on the current WorkloadManager, and
        samples new thresholds iid from the new WorkloadManager's distribution.

        Args:
            num_thresholds (int): Number of thresholds to sample for the new WorkloadManager.
            random_key (jnp.DeviceArray): JAX randomness key.

        Returns:
            WorkloadManager: New WorkloadManager object that is distributionally related to the
                current WorkloadManager.
        """
        # Initialize new WorkloadManager with same fields (except number of thresholds to sample)
        future_workload_manager = WorkloadManager(
            self.r,
            self.k,
            num_thresholds,
            self.onehot_transcoder,
            self.threshold_size_limit
        )
        future_workload_manager.filtered_thresholds = self.filtered_thresholds
        future_workload_manager.distribution_name = self.distribution_name
        future_workload_manager.distribution_parameters = self.distribution_parameters
        future_workload_manager.feature_probabilities = self.feature_probabilities
        future_workload_manager.threshold_sampling_probabilities = self.threshold_sampling_probabilities

        if self.distribution_name == "drift":
            gamma = future_workload_manager.distribution_parameters
            # Create canonical ordering (descending sort) of feature probabilities
            geometric_feature_probabilities = future_workload_manager.feature_probabilities
            sorted_indices = jnp.argsort(-geometric_feature_probabilities)
            sorted_probabilities = geometric_feature_probabilities[sorted_indices]
            # Generate keys associated with the sorted probabilities
            num_feats = len(geometric_feature_probabilities)
            keys = (1 - 2*gamma) * (jnp.arange(num_feats-1, -1, -1) / (num_feats - 1)) + \
                   (1-abs(1 - 2*gamma)) * random.uniform(random_key, (num_feats,))
            resorted_indices = jnp.argsort(-keys)
            resorted_probabilities = sorted_probabilities[resorted_indices]
            # Undo initial sorting on probabilities to obtain the new WorkloadManager's distribution probabilities
            future_workload_manager.feature_probabilities = resorted_probabilities[jnp.argsort(sorted_indices)]
            # Convert feature probabilities to threshold probabilities
            future_workload_manager.threshold_sampling_probabilities = \
                future_workload_manager._compute_threshold_probabilities_from_feature_probabilities(
                    future_workload_manager.filtered_thresholds
                )

        # Sample random thresholds for the future workload according to its distribution
        future_workload_manager._sample_random_thresholds(True, random_key)

        return future_workload_manager

    def initialize_random_thresholds(
        self,
        random_key: jnp.DeviceArray,
        distribution_name: str = "uniform",
        distribution_parameters: Optional[float] = None
    ) ->  None:
        """Initializes a concrete set of random thresholds according to the specified
        distribution, and enumerates and all queries consistent with them.

        Args:
            distribution (str, optional): Random distribution to select thresholds from. Defaults
                to "uniform".
        """
        self.distribution_name = distribution_name
        self.distribution_parameters = distribution_parameters

        # Generate all threshold combinations
        all_thresholds = itertools.combinations(
            range(self.onehot_transcoder.num_categorical_features),
            self.k
        )

        # Filter out all thresholds that have too many consistent queries
        self.filtered_thresholds = [
            threshold for threshold in all_thresholds
            if not self.threshold_size_limit
            or self.onehot_transcoder.get_domain_size(threshold) <= self.threshold_size_limit
        ]

        # For each threshold, compute its sampling probability from the given distribution
        # based on its individual features
        self.threshold_sampling_probabilities = self._compute_threshold_sampling_probabilities(
            distribution_name,
            distribution_parameters,
            self.filtered_thresholds,
            random_key
        )
        random_key, _ = random.split(random_key)

        self._sample_random_thresholds(False, random_key)


    def _sample_random_thresholds(
        self,
        with_replacement: bool,
        random_key: jnp.DeviceArray
    ):
        """Randomly samples thresholds based on the Workload's stored fields.

        Args:
            with_replacement (bool): Whether to sample with replacement (iid) or without.
            random_key (jnp.DeviceArray): JAX randomness key.
        """
        # Randomly sample subset of thresholds according to the sampling probabilities
        selected_thresholds_indices = jax.random.choice(
            random_key,
            len(self.filtered_thresholds),
            shape=(self.num_thresholds,),
            replace=with_replacement,
            p=self.threshold_sampling_probabilities
        )
        self.thresholds = [
            Threshold(
                self.r,
                self.k,
                self.filtered_thresholds[index],
                self.onehot_transcoder
            )
            for index in selected_thresholds_indices
        ]

        # Count the total number of consistent queries in the workload
        self.num_queries = sum([threshold.num_queries for threshold in self.thresholds])

        # Cache number of queries for fast local <--> global query ID lookup
        num_queries_by_threshold = jnp.array([
            threshold.num_queries
            for threshold in self.thresholds
        ])
        self._num_queries_by_threshold_cumsum = jnp.concatenate([
            jnp.array([0], dtype=jnp.int64),
            jnp.cumsum(num_queries_by_threshold, dtype=jnp.int64)
        ])
        assert self._num_queries_by_threshold_cumsum[-1] == self.num_queries

    def _compute_feature_probabilities(
        self,
        distribution_name: str,
        distribution_parameters: Optional[float] = None
    ) -> jnp.DeviceArray:
        """Computes the probabilities for sampling individual features from a given distribution.

        Args:
            distribution_name (str): Name of distribution being sampled from.
            distribution_parameters (Optional[float], optional): Parameter values of distribution
                 being sampled from. Defaults to None.

        Returns:
            jnp.DeviceArray: Probability values for each of the features.
        """
        num_features = self.onehot_transcoder.num_categorical_features
        feature_labels = range(1, num_features+1)
        if distribution_name == "uniform":
            pmf = stats.randint.pmf(
                feature_labels,
                low=1, 
                high=num_features+1
            )
        elif distribution_name == "zipf":
            pmf = stats.zipfian.pmf(
                feature_labels,
                a=distribution_parameters,
                n=num_features
            )
        elif distribution_name == "geometric":
            unnormalized_pmf = stats.geom.pmf(
                feature_labels,
                p=distribution_parameters
            )
            pmf = unnormalized_pmf / sum(unnormalized_pmf)
        elif distribution_name == "drift":
            # Hardcode Geometric(1/4) as base distribution
            unnormalized_pmf = stats.geom.pmf(
                feature_labels,
                p=1/4
            )
            pmf = unnormalized_pmf / sum(unnormalized_pmf)
        else:
            raise NotImplementedError(f"Distribution {distribution_name} not supported.")
        return pmf

    def _compute_threshold_sampling_probabilities(
        self,
        distribution_name: str,
        distribution_parameters: Optional[float],
        thresholds: List[Tuple[int, ...]],
        random_key: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """From the given feature distribution, computes the concrete sampling probabilities
        for each given threshold.

        Args:
            distribution_name (str): Name of distribution being sampled from.
            distribution_parameters (Optional[float]): Parameter values of distribution being
                sampled from.
            thresholds (List[Tuple[int, ...]]): Thresholds to compute sampling probabilities of.
            random_key (jnp.DeviceArray): JAX randomness key.

        Returns:
            jnp.DeviceArray: Array of probabilities corresponding to each threshold.
        """
        # Determine probability for each feature based on given distribution
        self.feature_probabilities = self._compute_feature_probabilities(
            distribution_name,
            distribution_parameters
        )
        # Shuffle probabilities to assign corresponding mass to a random feature (to remove
        # experimental bias)
        self.feature_probabilities = jax.random.permutation(random_key, self.feature_probabilities)
        random_key, _ = random.split(random_key)

        # Compute each threshold's sampling probability
        return self._compute_threshold_probabilities_from_feature_probabilities(thresholds)

    def _compute_threshold_probabilities_from_feature_probabilities(
        self,
        thresholds: List[Tuple[int, ...]]
    ) -> jnp.DeviceArray:
        """Converts the computed feature probabilities to threshold probabilities.

        Args:
            thresholds (List[Tuple[int, ...]]): Thresholds to compute sampling probabilities of.

        Returns:
            jnp.DeviceArray: Array of probabilities corresponding to each threshold.
        """
        jnp_threshold = jnp.array(thresholds, dtype=jnp.int32)
        def compute_threshold_probability(threshold):
            return self.feature_probabilities[threshold].prod()
        unnormalized_threshold_probabilities = vmap(
            compute_threshold_probability,
            in_axes=(0,)
        )(jnp_threshold)
        # Return sampling probabilities normalized over filtered thresholds
        return unnormalized_threshold_probabilities / unnormalized_threshold_probabilities.sum()

    def convert_global_ids_to_queries(
        self,
        global_ids: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Converts an array of global query IDs into an array of the corresponding queries.

        Args:
            global_ids (jnp.DeviceArray): Global IDs of the queries to be generated.

        Returns:
            jnp.DeviceArray: Queries corresponding to their global ID.
        """
        output, queries_threshold_ids, local_ids = self._generate_query_transformer_inputs(global_ids)
        max_intermediate_array_size = len(global_ids)
        for current_threshold_id in range(self.num_thresholds):
            output = self._transform_ids_to_queries(
                queries_threshold_ids,
                local_ids,
                current_threshold_id,
                max_intermediate_array_size,
                output
            )
        return output

    @partial(jit, static_argnums=(0,))
    def _generate_query_transformer_inputs(
        self,
        global_ids: jnp.DeviceArray
    ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]:
        """From an array of global IDs, this generates the relevant components used to transform
        the global IDs into their corresponding queries.

        Args:
            global_ids (jnp.DeviceArray): Global IDs of the queries to be generated.

        Returns:
            Tuple[jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray, jnp.DeviceArray]: An
                initialized array to store outputs, an array of the Threshold ID that each
                global ID corresponds to, and an array of the local IDs within the Threshold
                that each global ID corresponds to.
        """
        initialized_output = jnp.zeros(shape=(len(global_ids), self.k), dtype=jnp.uint32)
        queries_threshold_ids = jnp.searchsorted(
            self._num_queries_by_threshold_cumsum,
            global_ids,
            side="right"
        ).astype(jnp.uint32) - 1
        local_ids = vmap(self.threshold_id_and_global_id_to_local_id, in_axes=(0, 0))(
            queries_threshold_ids,
            global_ids
        )
        return initialized_output, queries_threshold_ids, local_ids

    def threshold_id_and_global_id_to_local_id(
        self,
        threshold_id: jnp.DeviceArray,
        global_id: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Given a Threshold ID and global ID of a query, determines the query's
        corresponding local ID.

        Args:
            threshold_id (jnp.DeviceArray): ID of the Threshold that the query corresponds to.
            global_id (jnp.DeviceArray): Global ID of the query.

        Returns:
            jnp.DeviceArray: Local ID of the query.
        """
        return global_id - self._num_queries_by_threshold_cumsum[threshold_id]

    @partial(jit, static_argnums=(0,3,4), donate_argnums=(5,))
    def _transform_ids_to_queries(
        self,
        queries_threshold_ids: jnp.DeviceArray,
        local_ids: jnp.DeviceArray,
        current_threshold_id: int,
        max_size: int,
        output: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Converts a set of implicitly represented queries corresponding to
        a specific Threshold into explicitly represented queries.

        Args:
            queries_threshold_ids (jnp.DeviceArray): Array containing each query's corresponding
                Threshold ID.
            local_ids (jnp.DeviceArray): Array containing each query's corresponding local ID.
            current_threshold_id (int): ID of the Threshold currently being evaluated.
            max_size (int): Largest number of queries that could correspond to the current
                Threshold.
            output (jnp.DeviceArray): Array containing the final shared output of this Threshold's
                computation + all other Thresholds' computations.

        Returns:
            jnp.DeviceArray: The final output of this Threshold's computation (a shared array
                with all other Thresholds' computations).
        """
        relevant_indices = jnp.nonzero(
            queries_threshold_ids == current_threshold_id,
            size=max_size,
            fill_value=2147483647
        )[0]
        relevant_local_ids = local_ids.at[relevant_indices].get(mode='drop', indices_are_sorted=True, unique_indices=True)
        queries = self.thresholds[current_threshold_id].local_ids_to_queries(relevant_local_ids)
        output = output.at[relevant_indices].set(queries, mode='drop')
        return output

    def answer_queries_of_all_thresholds(
        self,
        dataset: jnp.DeviceArray,
        exclude_indices: Optional[Dict[Threshold, jnp.DeviceArray]] = None
    ) -> Dict[Threshold, jnp.DeviceArray]:
        """Answers each threshold's queries directly on categorical data.

        Args:
            dataset (jnp.DeviceArray): Dataset used to answer the queries.
            exclude_indices (Optional[Dict[Threshold, jnp.DeviceArray]], optional): For each
                threshold, indices for which query answers to exclude from the final result.
                Defaults to None.

        Returns:
            Dict[Threshold, jnp.DeviceArray]: Answers to the queries of each threshold.
        """
        return {
            threshold: self.answer_queries_of_threshold(
                threshold,
                dataset,
                exclude_indices[threshold] if exclude_indices is not None else None
            )
            for threshold in self.thresholds
        }

    def answer_queries_of_threshold(
        self,
        threshold: Threshold,
        dataset: jnp.DeviceArray,
        exclude_indices: Optional[jnp.DeviceArray] = None
    ) -> jnp.DeviceArray:
        """Answers each threshold's queries directly on categorical data.

        Args:
            threshold (Threshold): Threshold whose queries are to be answered.
            dataset (jnp.DeviceArray): Dataset used to answer the queries.
            exclude_indices (Optional[Dict[Threshold, jnp.DeviceArray]], optional): For each
                threshold, indices for which query answers to exclude from the final result.
                Defaults to None.

        Returns:
            Dict[Threshold, jnp.DeviceArray]: Answers to the queries of each threshold.
        """
        answers = threshold.evaluate_consistent_queries(dataset)
        if exclude_indices is not None:
            answers = jnp.delete(answers, exclude_indices, axis=0)
        return answers

    def evaluate_categorical_predicate(
        self,
        categorical_features_and_categories: jnp.DeviceArray,
        dataset_row: jnp.DeviceArray
    ) -> jnp.bool_:
        """Evaluates an r-of-k threshold predicate directly on a row of categorical data.

        Args:
            categorical_features_and_categories (jnp.DeviceArray): (2,k) vector of the k-features
                to operate over in the dataset row and their specific categorical values.
            dataset_row (jnp.DeviceArray): k-element array of categorical values.

        Returns:
            jnp.bool_: Result of the r-of-k threshold predicate.
        """
        features, categories = categorical_features_and_categories
        return self.categorical_predicate_evaluator(
            categories,
            dataset_row[features]
        )

    def evaluate_surrogate_predicate(
        self,
        predicate_indices: jnp.DeviceArray,
        dataset_row: jnp.DeviceArray
    ) -> jnp.float32:
        """Evaluates an r-of-k threshold predicate directly on a row of onehot data.

        Args:
            r (int): Number of elements that have to match between the predicate vector
                and dataset row.
            predicate_indices (jnp.DeviceArray): k-element array of onehot values.
            dataset_row (jnp.DeviceArray): k-element array of onehot values.

        Returns:
            jnp.float32: Result of the r-of-k threshold predicate.
        """
        return self.surrogate_predicate_evaluator(dataset_row[predicate_indices])

    @partial(jit, static_argnums=(0,))
    def answer_surrogate_queries(
        self,
        queries: jnp.DeviceArray,
        dataset: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Answers a given subset of surrogate queries across the entire dataset.

        Args:
            queries (jnp.DeviceArray): Subset of surrogate queries to be answered.
            dataset (jnp.DeviceArray): Dataset used to answer the queries.

        Returns:
            jnp.DeviceArray: Answers to the queries.
        """
        evaluate_predicates_on_dataset = vmap(
            vmap(
                self.evaluate_surrogate_predicate,
                in_axes=(0, None)
            ),
            in_axes=(None, 0)
        )
        return evaluate_predicates_on_dataset(queries, dataset).mean(axis=0)

    def answer_categorical_queries_of_all_thresholds(
        self,
        dataset: jnp.DeviceArray,
        exclude_indices: Optional[Dict[Threshold, jnp.DeviceArray]] = None
    ) -> Dict[Threshold, jnp.DeviceArray]:
        """Answers each threshold's queries directly on categorical data.

        Args:
            dataset (jnp.DeviceArray): Dataset used to answer the queries.
            exclude_indices (Optional[Dict[Threshold, jnp.DeviceArray]], optional): For each
                threshold, indices for which query answers to exclude from the final result.
                Defaults to None.

        Returns:
            Dict[Threshold, jnp.DeviceArray]: Answers to the queries of each threshold.
        """
        return {
            threshold: self.answer_categorical_queries_of_threshold(
                threshold,
                dataset,
                exclude_indices[threshold] if exclude_indices is not None else None
            )
            for threshold in self.thresholds
        }

    def answer_categorical_queries_of_threshold(
        self,
        threshold: Threshold,
        dataset: jnp.DeviceArray,
        exclude_indices: Optional[jnp.DeviceArray] = None
    ) -> jnp.DeviceArray:
        """Answers a given threshold's categorical queries across the entire dataset.

        Args:
            threshold (Threshold): Threshold whose queries are to be answered.
            dataset (jnp.DeviceArray): Dataset used to answer the queries.
            exclude_indices (Optional[jnp.DeviceArray], optional): Indices for which query answers
                to exclude from the final result. Defaults to None.

        Returns:
            jnp.DeviceArray: Answers to the queries.
        """
        queries = threshold.generate_all_categorical_queries(prepend_features=True)
        answers = self._answer_categorical_queries(
            queries,
            dataset
        )
        if exclude_indices is not None:
            answers = jnp.delete(answers, exclude_indices, axis=0)
        return answers

    def _answer_categorical_queries(
        self,
        queries: jnp.DeviceArray,
        dataset: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Answers a given subset of categorical queries across the entire dataset.

        Args:
            queries (jnp.DeviceArray): Subset of categorical queries to be answered.
            dataset (jnp.DeviceArray): Dataset used to answer the queries.

        Returns:
            jnp.DeviceArray: Answers to the queries.
        """
        return self._answer_queries(
            queries,
            dataset,
            self.evaluate_categorical_predicate,
            128
        )

    def answer_surrogate_queries_of_all_thresholds(
        self,
        dataset: jnp.DeviceArray,
        exclude_indices: Optional[Dict[Threshold, jnp.DeviceArray]] = None
    ) -> Dict[Threshold, jnp.DeviceArray]:
        """Answers each threshold's surrogate queries on relaxed onehot data.

        Args:
            dataset (jnp.DeviceArray): Dataset used to answer the queries.
            exclude_indices (Optional[Dict[Threshold, jnp.DeviceArray]], optional): For each
                threshold, indices for which query answers to exclude from the final result.
                Defaults to None.

        Returns:
            Dict[Threshold, jnp.DeviceArray]: Answers to the queries of each threshold.
        """
        return {
            threshold: self.answer_surrogate_queries_of_threshold(
                threshold,
                dataset,
                exclude_indices[threshold] if exclude_indices is not None else None
            )
            for threshold in self.thresholds
        }

    def answer_surrogate_queries_of_threshold(
        self,
        threshold: Threshold,
        dataset: jnp.DeviceArray,
        exclude_indices: Optional[jnp.DeviceArray] = None
    ) -> jnp.DeviceArray:
        """Answers a given threshold's surrogate queries across the entire dataset.

        Args:
            threshold (Threshold): Threshold whose queries are to be answered.
            dataset (jnp.DeviceArray): Dataset used to answer the queries.
            exclude_indices (Optional[jnp.DeviceArray], optional): Indices for which query answers
                to exclude from the final result. Defaults to None.

        Returns:
            jnp.DeviceArray: Answers to the queries.
        """
        queries = threshold.generate_all_onehot_queries()
        answers = self._answer_surrogate_queries(
            queries,
            dataset
        )
        if exclude_indices is not None:
            answers = jnp.delete(answers, exclude_indices, axis=0)
        return answers

    def _answer_surrogate_queries(
        self,
        queries: jnp.DeviceArray,
        dataset: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Answers a given subset of surrogate queries across the entire dataset.

        Args:
            queries (jnp.DeviceArray): Subset of surrogate queries to be answered.
            dataset (jnp.DeviceArray): Dataset used to answer the queries.

        Returns:
            jnp.DeviceArray: Answers to the queries.
        """
        return self._answer_queries(
            queries,
            dataset,
            self.evaluate_surrogate_predicate,
            128
        )

    @partial(jit, static_argnums=(0,3,4,5))
    def _answer_queries(
        self,
        queries: jnp.DeviceArray,
        dataset: jnp.DeviceArray,
        predicate_fn: Callable,
        query_set_size_fudge_factor: int
    ) -> jnp.DeviceArray:
        """Answers an arbitrary set of queries across the entire dataset.

        Args:
            queries (jnp.DeviceArray): Subset of surrogate queries to be answered.
            dataset (jnp.DeviceArray): Dataset used to answer the queries.
            predicate_fn (Callable): Function used to evaluate a single predicate on a single
                row of data.
            max_query_set_size (int): Size of the largest query set to answer in parallel.

        Returns:
            jnp.DeviceArray: Answers to the queries.
        """
        # Create function to answer all predicates on a single dataset row
        evaluate_predicates_on_row = vmap(
            predicate_fn,
            in_axes=(0, None)
        )

        # Heuristic value for "largest query size" to evaluate in parallel
        max_query_set_size = int(self.total_device_memory // \
            (query_set_size_fudge_factor * self.k * binom(self.k, self.r)))

        # Split query set into chunks, and evaluate each sequentially
        num_chunks = math.ceil(len(queries) / max_query_set_size)
        chunked_queries = jnp.array_split(queries, num_chunks)

        # Create an aggregation function for the scan operation
        def sum_all_predicate_evaluations(predicates, prior_result, data_row):
            sum_result = jnp.add(
                prior_result,
                evaluate_predicates_on_row(predicates, data_row),
            )
            return sum_result, None

        chunked_answers = []
        for chunk_of_queries in chunked_queries:
            # Reduce the inputs to the function to be only the dataset row
            sum_predicate_chunk_evaluations_on_row = jax.tree_util.Partial(
                sum_all_predicate_evaluations,
                chunk_of_queries
            )

            # Scan over rows in dataset, answering queries on each row and accumulating the results
            init_value = jnp.zeros(len(chunk_of_queries), dtype=jnp.float32)
            query_answers, _ = lax.scan(
                sum_predicate_chunk_evaluations_on_row,
                init_value,
                dataset,
                unroll=1
            )
            chunked_answers.append(query_answers)

        return jnp.concatenate(chunked_answers) / len(dataset)
