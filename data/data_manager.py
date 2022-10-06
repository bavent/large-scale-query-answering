"""Performs all high level management of a particular dataset."""

import csv
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp

from data.onehot_transcoder import OnehotTranscoder


class DataManager:
    """Manages the importing and storage of a dataset in various usable forms."""

    def __init__(
        self,
        dataset_name: str,
        load_num_rows: Optional[int] = None
    ) -> None:
        """Performs all primary functions to initialize the specified dataset, the onehot encoder,
         and the onehot encoded dataset.

        Args:
            dataset_name (str): Name of the dataset ('ADULT' or 'LOANS').
        """
        self.dataset_name = str.lower(dataset_name)
        if self.dataset_name not in {'adult', 'loans'}:
            raise NotImplementedError(f"Processing for {dataset_name} dataset not implemented.")

        self._data_path_prefix = f'data/{self.dataset_name}'
        self._file_paths = {}

        self._load_data_from_csv(load_num_rows)
        self._generate_onehot_transcoder()
        self._generate_onehot_data()

    def _load_data_from_csv(self, load_num_rows: Optional[int] = None) -> None:
        """Imports dataset from a CSV file.
        """
        self._file_paths['categorical_data_path'] = f'{self._data_path_prefix}/categorical_data.csv'
        self._file_paths['categorical_json_path'] = f'{self._data_path_prefix}/domain.json'

        # Get features from dataset
        with open(
            self._file_paths['categorical_data_path'], 'r', encoding='utf8', newline=''
        ) as data_file:
            self.feature_names = next(csv.reader(data_file))
            self.num_categorical_features = len(self.feature_names)

        # Get actual data from file and format into ndarray
        self.categorical_data = jnp.array(onp.genfromtxt(
            self._file_paths['categorical_data_path'],
            dtype=int,
            delimiter=',',
            skip_header=True
        ))
        if load_num_rows and load_num_rows < len(self.categorical_data):
            self.categorical_data = self.categorical_data[:load_num_rows,:]
        self.num_rows = len(self.categorical_data)

    def _generate_onehot_transcoder(self) -> None:
        """Generates a onehot transcoder for the given data.
        """
        self.onehot_transcoder = OnehotTranscoder(
            self.categorical_data,
            self.feature_names,
            self._file_paths['categorical_json_path']
        )

    def _generate_onehot_data(self) -> None:
        """Uses the onehot encoder to generate a onehot encoded version of the categorical dataset.
        """
        self.onehot_data = self.onehot_transcoder.onehot_encode_dataset(self.categorical_data)
        self.num_onehot_features = jnp.size(self.onehot_data, 1)

    def generate_random_relaxed_dataset(
        self,
        num_rows: int,
        random_key: jnp.DeviceArray,
    ) -> jnp.DeviceArray:
        """Generates a uniformly random relaxed variant of the underlying dataset.

        Args:
            num_rows (int): Number of rows that the relaxed dataset should have.
            random_key (jnp.DeviceArray): JAX randomness key.
        """
        relaxed_dataset_shape = (num_rows, self.num_onehot_features)
        return random.uniform(
            random_key,
            relaxed_dataset_shape,
            minval=0,
            maxval=1,
            dtype=jnp.float32
        )

    def check_transcoder_sanity(self, check_manual_transcoding: bool = True) -> bool:
        """Checks if encoding + decoding the categorical data yields the original categorical data.

        Args:
            check_manual_transcoding (bool, optional): Checks if manually encoding + decoding
                data matches original data. Warning: Very slow. Defaults to True.

        Returns:
            bool: Returns True iff the encoded + decoded data matches the original data.
        """
        print("Checking transcoder sanity...", end="")
        auto_encoded_data = self.onehot_data
        transcoder = self.onehot_transcoder

        # Check that encoding then decoding yields original data
        auto_decoded_data = transcoder.onehot_decode_dataset(auto_encoded_data)
        auto_transcoding_stable = jnp.alltrue(self.categorical_data == auto_decoded_data)

        manual_transcoding_stable = True
        if check_manual_transcoding:
            # Check that manual encoding/decoding matches auto encoding/decoding
            manual_encoded_data = transcoder.onehot_encode_dataset_manual(self.categorical_data)
            stable_manual_encoding = jnp.alltrue(manual_encoded_data == auto_encoded_data)
            manual_decoded_data = transcoder.onehot_decode_dataset_manual(manual_encoded_data)
            stable_manual_decoding = jnp.alltrue(manual_decoded_data == auto_decoded_data)
            manual_transcoding_stable = stable_manual_encoding and stable_manual_decoding

        result = bool(auto_transcoding_stable and manual_transcoding_stable)
        print(f"{'PASSED' if result else 'FAILED'}!")
        return result


if __name__ == '__main__':
    pass
