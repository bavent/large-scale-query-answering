"""Provides encoding/decoding capabilities between categorical and onehot data representations."""

import json
from typing import List, Tuple

import jax.nn as jnn
import jax.numpy as jnp
from jax import jit, vmap
from sklearn import preprocessing


class OnehotTranscoder:
    """Encoding and decoding capabilities for a categorical to/from a onehot dataset."""

    def __init__(
        self,
        categorical_data: jnp.DeviceArray,
        feature_names: List[str],
        categorical_json_path: str
    ) -> None:
        """Initializes a OnehotTranscoder object from a JSON file.

        Args:
            categorical_data (jnp.DeviceArray): Categorical dataset to build transcoder.
            feature_names (List[str]): Names of features in order of categorical data columns.
            categorical_json_path (str): Path to JSON file containing dataset domain info.
        """
        self._load_data_domain_from_json(feature_names, categorical_json_path)
        self._generate_datapoint_transcoder(categorical_data)
        self._generate_feature_transcoders()

    def _load_data_domain_from_json(
        self,
        feature_names: List[str],
        categorical_json_path: str
    ) -> None:
        """Import domain information for dataset from a JSON file.

        Args:
            categorical_json_path (str): Path to JSON file containing dataset domain info.
        """
        # Get domain info for each feature
        with open(categorical_json_path, 'r', encoding='utf8') as json_file:
            self._data_domain = json.load(json_file)

        self.num_categorical_features = len(self._data_domain)
        self.num_onehot_features = sum(self._data_domain.values())

        # Enumerate domain info for each feature
        self._domain_values = []
        for feature_name in feature_names:
            feature_domain = self._data_domain[feature_name]
            self._domain_values.append(jnp.arange(feature_domain))

    def _generate_datapoint_transcoder(self, categorical_data: jnp.DeviceArray) -> None:
        """Onehot encode the entire dataset using previously-generated domain values.

        Args:
            categorical_data (jnp.DeviceArray): Dataset in its categorical representation.
        """
        self._datapoint_transcoder = preprocessing.OneHotEncoder(
            categories=self._domain_values,
            sparse=False,
            dtype=int
        ).fit(categorical_data)

    def _generate_feature_transcoders(self) -> None:
        """Onehot encode each individual feature using previously-generated domain values.
        """
        feature_onehot_index_range = []
        current_index = 0
        for categories in self._domain_values:
            # Define index range for each feature's onehot representation within the datapoint
            start_index = current_index
            current_index = start_index + len(categories)
            feature_onehot_index_range.append((start_index, current_index))
        self.feature_onehot_index_range = tuple(feature_onehot_index_range)

    def get_categories(self, feature: int) -> jnp.DeviceArray:
        """Gets all the categories of a given feature.

        Args:
            feature (int): The feature to get the categories of.

        Returns:
            jnp.DeviceArray: The categories of the given feature.
        """
        return self._domain_values[feature]

    def get_domain_size(self, features: Tuple[int]) -> int:
        """Gets the size of the domain of a product of features.

        Args:
            features (Tuple[int]): Set of features to get the domain size of.

        Returns:
            int: Domain size of features.
        """
        product = 1
        for feature in features:
            product *= len(self._domain_values[feature])
        return product

    def get_global_hot_index_of_feature_category(self, feature_num: int, category: int) -> int:
        """Given a particular category of a specific feature, find which index is 'hot'
        in the entire datapoint's onehot encoding.

        Args:
            feature_num (int): The number representing a particular feature (e.g., age).
            category (int): The number representing a particular value of the
                category (e.g., 30 if the age is 30).

        Returns:
            int: Within the datapoint's onehot encoding of the particular feature, returns which
                index is 1.
        """
        return category + self.feature_onehot_index_range[feature_num][0]

    def onehot_encode_datapoint(self, datapoint: jnp.DeviceArray) -> jnp.DeviceArray:
        """Onehot encodes all features of a single datapoint.

        Args:
            datapoint (jnp.DeviceArray): Categorical datapoint to be onehot encoded.

        Returns:
            jnp.DeviceArray: Onehot encoding of categorical datapoint.
        """
        onehot_encoding = []
        for feature_index in range(self.num_categorical_features):
            feature_onehot_encoding = jnn.one_hot(
                datapoint[feature_index],
                len(self._domain_values[feature_index]),
                dtype=jnp.float32
            )
            onehot_encoding.append(feature_onehot_encoding)
        return jnp.concatenate(onehot_encoding)

    def onehot_decode_datapoint(self, datapoint: jnp.DeviceArray) -> jnp.DeviceArray:
        """Onehot decodes all features of a single datapoint.

        Args:
            datapoint (jnp.DeviceArray): Onehot encoded datapoint to be decoded.

        Returns:
            jnp.DeviceArray: Categorical encoding of onehot datapoint.
        """
        categorical_feature_encoding = []
        for feature_index in range(self.num_categorical_features):
            start_index, end_index = self.feature_onehot_index_range[feature_index]
            categorical_feature_encoding.append(datapoint[start_index:end_index].argmax())
        return jnp.array(categorical_feature_encoding)

    def onehot_encode_dataset_manual(self, categorical_data: jnp.DeviceArray) -> jnp.DeviceArray:
        """Onehot encodes entire dataset.

        Args:
            categorical_data (jnp.DeviceArray): Categorical dataset to be onehot encoded.

        Returns:
            jnp.DeviceArray: Onehot encoding of entire categorical dataset.
        """
        return vmap(jit(self.onehot_encode_datapoint))(categorical_data)

    def onehot_decode_dataset_manual(self, onehot_data: jnp.DeviceArray) -> jnp.DeviceArray:
        """Onehot decodes entire dataset.

        Args:
            onehot_data (jnp.DeviceArray): Onehot encoded dataset to be decoded.

        Returns:
            jnp.DeviceArray: Categorical encoding of the onehot dataset.
        """
        return vmap(jit(self.onehot_decode_datapoint))(onehot_data)

    def onehot_encode_dataset(self, categorical_data: jnp.DeviceArray) -> jnp.DeviceArray:
        """Onehot encodes entire dataset.

        Args:
            categorical_data (jnp.DeviceArray): Categorical dataset to be onehot encoded.

        Returns:
            jnp.DeviceArray: Onehot encoding of entire categorical dataset.
        """
        return jnp.array(
            self._datapoint_transcoder.transform(categorical_data),
            dtype=jnp.float32
        )

    def onehot_decode_dataset(self, onehot_data: jnp.DeviceArray) -> jnp.DeviceArray:
        """Onehot decodes entire dataset.

        Args:
            onehot_data (jnp.DeviceArray): Onehot encoded dataset to be decoded.

        Returns:
            jnp.DeviceArray: Categorical encoding of the onehot dataset.
        """
        return self._datapoint_transcoder.inverse_transform(onehot_data)
