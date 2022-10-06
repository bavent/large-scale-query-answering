"""Performs all high level management of r-of-k thresholds."""

import itertools
import string
from functools import partial
from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
from jax import jit, vmap
from scipy.special import comb

from data.onehot_transcoder import OnehotTranscoder


class Threshold:
    """Represents an r-of-k threshold and its consistent queries, and provides static
     functionality to manipulate and answer them."""

    def __init__(
        self,
        r: int,
        k: int,
        categorical_features: Tuple,
        onehot_transcoder: OnehotTranscoder,
    ):
        """Initializes the threshold with its consistent queries.

        Args:
            r (int): Parameter specifying the number of matching categories needed to surpass the
                threshold for any query.
            k (int): Parameter specifying the number of categorical features in the threshold.
            categorical_features (jnp.DeviceArray): The concrete categorical features that compose
                the threshold.
            onehot_transcoder (OnehotTranscoder): Transcoder used to move between categorical and
                onehot representations of the data space.
        """
        self.r = r
        self.k = k
        self.categorical_features = categorical_features
        self.onehot_transcoder = onehot_transcoder
        self.categories = [
            onehot_transcoder.get_categories(feature)
            for feature in categorical_features
        ]
        self.onehot_indices = self.generate_compact_onehot_indices()
        self.num_queries = onehot_transcoder.get_domain_size(categorical_features)
        self.evaluate_consistent_queries = self.generate_threshold_consistent_queries_evaluator()

    def generate_compact_onehot_indices(self) -> List:
        """From the Threshold's features and categories, generates a compact representation of all
        corresponding onehot query indices.

        Returns:
            List: A k-element list, where each element corresponds to all onehot indices of a
                feature. The cartesian product of all k elements would be equivalent to enumerating
                all consistent onehot queries of this Threshold.
        """
        feature_categories_to_onehot_indices = vmap(
            self.onehot_transcoder.get_global_hot_index_of_feature_category,
            in_axes=(None, 0)
        )
        onehot_indices = []
        for i, feature in enumerate(self.categorical_features):
            categories = self.categories[i]
            onehot_indices.append(feature_categories_to_onehot_indices(feature, categories))
        return onehot_indices

    @partial(jit, static_argnums=(0,))
    def local_ids_to_queries(
        self,
        local_indices: jnp.DeviceArray
    ) -> jnp.DeviceArray:
        """Converts a set of this threshold's indices which implicity represent a subset of
        queries explicitly into those queries.

        Args:
            local_indices (jnp.DeviceArray): Indices of the queries implicitly being represented.

        Returns:
            jnp.DeviceArray: Array of queries represented as the onehot indices that they evaluate.
        """
        # Convert local indices into local categories
        categories_shape = tuple(len(categories) for categories in self.categories)
        threshold_categories = jnp.array(
            jnp.unravel_index(local_indices, categories_shape)
        ).T.reshape(-1, self.k)

        # Convert threshold feature + local categories into global onehot indices (i.e., query)
        feature_categories_to_onehot_indices = vmap(
            self.onehot_transcoder.get_global_hot_index_of_feature_category,
            in_axes=(None, 0)
        )
        onehot_queries = []
        for i, feature in enumerate(self.categorical_features):
            categories = threshold_categories[:,i]
            onehot_queries.append(feature_categories_to_onehot_indices(feature, categories))
        return jnp.column_stack(onehot_queries)

    @partial(jit, static_argnums=(0,1))
    def generate_all_categorical_queries(
        self,
        prepend_features: Optional[bool] = False
    ) -> jnp.DeviceArray:
        """Enumerates all queries consistent with the threshold.

        Args:
            prepend_features (Optional[bool], optional): Prepend the threshold's features to the
                queries being generated (useful if directly evaluating the categorical queries).
                Defaults to False.

        Returns:
            jnp.DeviceArray: All consistent queries, represented as all combinations of the k
                features' categories.
        """
        categorical_queries = jnp.stack(
            jnp.meshgrid(*self.categories, indexing='ij'),
            axis=-1
        ).reshape(-1, self.k)

        if prepend_features:
            # Attach features to the corresponding categories on new axis
            categorical_queries = jnp.stack([
                jnp.broadcast_to(
                    self.categorical_features,
                    categorical_queries.shape
                ),
                categorical_queries,
            ],
            axis=1)

        return categorical_queries

    @partial(jit, static_argnums=(0,))
    def generate_all_onehot_queries(self) -> jnp.DeviceArray:
        """Generates the set of queries in onehot representation by first generating the
            set of all consistent categorical queries and then converting.

        Returns:
            jnp.DeviceArray: All consistent queries, represented as the onehot indices of
                all combinations of the k features' categories.
        """
        def generate_onehot_query(categorical_query):
            onehot_query = []
            for j in range(self.k):
                feature = self.categorical_features[j]
                category = categorical_query[j]
                hot_index = self.onehot_transcoder.get_global_hot_index_of_feature_category(
                    feature,
                    category
                )
                onehot_query.append(hot_index)
            return jnp.array(onehot_query)

        categorical_queries = self.generate_all_categorical_queries()

        return vmap(jit(generate_onehot_query))(categorical_queries)

    def generate_threshold_consistent_queries_evaluator(
        self
    ) -> Callable[[jnp.DeviceArray], jnp.DeviceArray]:
        """Generates a predicate answering function for all queries consistent with any
        r-of-k threshold.

        Returns:
            Callable[[jnp.DeviceArray], jnp.DeviceArray]: Function to answer all consistent
                queries of this threshold.
        """
        r = self.r
        k = self.k

        # Construct string for k-th order statistic with einsum
        letters_idx = string.ascii_lowercase
        row_idx, col_idxs = letters_idx[0], letters_idx[1:]
        einsum_string = ",".join([row_idx + a for a in col_idxs[:k]]) + "->" + col_idxs[:k]

        # When r is less than k/2, we can evaluate fewer monomial terms by computing polynomial's
        # negation on negated datapoints
        compute_negation = r <= k/2
        if compute_negation:
            r = k-r+1

        # Generates all k-length mask combinations to represent exactly i positive terms and the
        # corresponding k-i negated terms in the r-of-k threshold
        mask_combos = []
        for i in range(r, k+1):
            base_mask = [False]*(k-i) + [True]*i
            i_length_mask_combos = list(set(itertools.permutations(base_mask)))
            mask_combos.extend(i_length_mask_combos)

        # Returns positive/negated dataset columns of the threshold's features based on mask values.
        def extract_projected_data(dataset, mask_combo):
            projected_data = []
            for i, mask in enumerate(mask_combo):
                projected_feature = dataset[:, self.onehot_indices[i]]
                if mask:
                    projected_data.append(projected_feature)
                else:
                    projected_data.append(1-projected_feature)
            return projected_data

        # Computes the positive product queries of threshold's features (with some terms
        # potentially negated)
        def evaluate_all_positive_product_queries(projected_data, einsum_string):
            return jnp.einsum(einsum_string, *projected_data).flatten()

        # Blunt evaluator function which simply computes the sum of all combinations of >= r
        # positive feature values (with the remaining <= k-r feature values negated)
        def evaluate_all_threshold_queries(dataset):
            result = jnp.zeros((self.num_queries,), dtype=jnp.float32)
            for mask_combo in mask_combos:
                projected_data = extract_projected_data(dataset, mask_combo)
                product_queries_result = evaluate_all_positive_product_queries(
                    projected_data,
                    einsum_string
                )
                result += product_queries_result
            return result / len(dataset)

        # Return polynomial evaluator (or its negation on negated data)
        if compute_negation:
            return jit(lambda x: 1 - evaluate_all_threshold_queries(1 - x))
        else:
            return jit(evaluate_all_threshold_queries)

    @staticmethod
    def generate_categorical_predicate_evaluator(
        r: int,
        k: int
    ) -> Callable[[jnp.DeviceArray, jnp.DeviceArray], jnp.bool_]:
        """Generates a categorical predicate answering function for an r-of-k threshold.

        Args:
            r (int): Number of elements that have to match between the predicate vector
                and dataset row.
            k (int): Number of elements in the predicate vector.

        Returns:
            Callable[jnp.DeviceArray, jnp.DeviceArray], jnp.bool_]: Function to
                answer a given  categorical predicate on a given datapoint.
        """
        def evaluate_categorical_predicate(
            predicate_categories: jnp.DeviceArray,
            data_elements: jnp.DeviceArray
        ) -> jnp.DeviceArray:
            return jnp.count_nonzero(predicate_categories == data_elements) >= r

        return jit(evaluate_categorical_predicate)

    @staticmethod
    def _inclusion_exclusion_surrogate_predicate_evaluator(
        r: int,
        k: int
    ) -> Callable[[jnp.DeviceArray], jnp.float32]:
        """Generates a surrogate predicate answering function for this r-of-k threshold using an
        inclusion-exclusion principle approach.

        Args:
            r (int): Number of elements that have to match between the predicate vector
                and dataset row.
            k (int): Number of elements in the predicate vector.

        Returns:
            Callable[[jnp.DeviceArray], jnp.float32]: Function to answer a given
                surrogate predicate on a given datapoint.
        """
        # When r is less than k/2, we can evaluate fewer monomial terms by computing polynomial's
        # negation on negated datapoints
        compute_negation = r <= k/2
        if compute_negation:
            r = k-r+1

        # Generates all k-length mask combinations to represent exactly i positive terms and the
        # corresponding k-i negated terms in the r-of-k threshold
        def generate_masks(i: int, k: int) -> jnp.DeviceArray:
            base_mask = [False]*(k-i) + [True]*i
            return jnp.array(list(set(itertools.permutations(base_mask))))

        # Generate all r...k-length mask combinations for each monomial term, and generate
        # each term's corresponding binomial coefficient in the inclusion-exclusion expression
        all_masks = []
        binomial_coefficients = []
        current_sign = 1
        for i in range(r, k+1):
            current_masks = generate_masks(i, k)
            all_masks.extend(current_masks)
            binomial_coefficients.extend(
                current_sign * comb(i-1, i-r, exact=True) * jnp.ones(len(current_masks), dtype=jnp.int32)
            )
            current_sign *= -1
        all_masks = jnp.array(all_masks, dtype=jnp.bool_)
        binomial_coefficients = jnp.array(binomial_coefficients, dtype=jnp.int16)

        # Computes the (monomial) product predicate of 1-masked data elements, multiplies by its
        # corresponding binomial coefficient. Vmap to compute all predicates simultaneously.
        def positive_product_predicate(elements, mask, binomial_coefficient):
            return jnp.where(mask == True, elements, 1).prod() * binomial_coefficient
        all_positive_product_predicates = \
            vmap(jit(positive_product_predicate), in_axes=(None, 0, 0))

        # Evaluates polynomial threshold surrogate predicate by summing all product predicate
        # monomial terms.
        def evaluate_polynomial_threshold(data_elements: jnp.DeviceArray) -> jnp.DeviceArray:
            return jnp.sum(
                all_positive_product_predicates(data_elements, all_masks, binomial_coefficients)
            )

        # Return polynomial evaluator (or its negation on negated data)
        if compute_negation:
            return jit(lambda x: 1 - evaluate_polynomial_threshold(1 - x))
        else:
            return jit(evaluate_polynomial_threshold)

    @staticmethod
    def _blunt_surrogate_predicate_evaluator(
        r: int,
        k: int
    ) -> Callable[[jnp.DeviceArray], jnp.float32]:
        """Generates a surrogate predicate answering function for this r-of-k threshold using a
        blunt 'all product' combinations approach.

        Args:
            r (int): Number of elements that have to match between the predicate vector
                and dataset row.
            k (int): Number of elements in the predicate vector.

        Returns:
            Callable[[jnp.DeviceArray], jnp.float32]: Function to answer a given
                surrogate predicate on a given datapoint.
        """
        # Generates all k-length mask combinations to represent exactly i positive terms and the
        # corresponding k-i negated terms in the r-of-k threshold
        def generate_masks(i: int, k: int) -> jnp.DeviceArray:
            base_mask = [False]*(k-i) + [True]*i
            return jnp.array(list(set(itertools.permutations(base_mask))))

        # Generate all k-length mask combinations to represent >= r positive terms and the
        # corresponding < k-r negated terms in the r-of-k threshold
        all_masks = []
        for i in range(r, k+1):
            all_masks.extend(generate_masks(i, k))
        all_masks = jnp.array(all_masks, dtype=jnp.bool_)

        # Define the product predicate, which multiplies each element or its negation, according
        # to the mask. Then vmap over all masks.
        def product_predicate(elements: jnp.DeviceArray, mask: jnp.DeviceArray) -> jnp.DeviceArray:
            return jnp.where(mask==True, elements, 1).prod() * \
                   jnp.where(mask==False, 1-elements, 1).prod()
        all_product_predicates = vmap(jit(product_predicate), in_axes=(None, 0))

        # Define function to evaluate then sum all product predicates over every mask combination.
        def evaluate_polynomial_threshold(data_elements: jnp.DeviceArray) -> jnp.DeviceArray:
            return jnp.sum(
                all_product_predicates(data_elements, all_masks)
            )

        return jit(evaluate_polynomial_threshold)

    @staticmethod
    def generate_surrogate_predicate_evaluator(
        r: int,
        k: int
    ) -> Callable[[jnp.DeviceArray], jnp.float32]:
        """Generates a surrogate predicate answering function for this r-of-k threshold.

        Args:
            r (int): Number of elements that have to match between the predicate vector
                and dataset row.
            k (int): Number of elements in the predicate vector.

        Returns:
            Callable[[jnp.DeviceArray], jnp.float32]: Function to answer a given
                surrogate predicate on a given datapoint.
        """
        return Threshold._inclusion_exclusion_surrogate_predicate_evaluator(r, k)

    def __lt__(
        self,
        other: 'Threshold'
    ) -> bool:
        """Defines the canonical ordering of Thresholds as the ordering of their features.

        Args:
            other (Threshold): Threshold to compare against.

        Returns:
            bool: Whether current Theshold is less than other Threshold.
        """
        return self.categorical_features < other.categorical_features

    def __str__(self) -> str:
        """Defining name of Threshold object to be represented as its underlying features.

        Returns:
            str: Name of the Threshold object.
        """
        return "Threshold"+str(self.categorical_features)
