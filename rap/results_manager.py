"""Functionality for computing and storing baseline mechanism results, as well as RAP results
throughout RP iterations."""

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import dill
import jax.numpy as jnp
import jax.random as random

from rap.dp_mechanisms import gaussian
from rap.optimization_hyperparameters import OptimizationHyperparameters
from rap.rap_inputs import RAPInputs
from rap.threshold import Threshold
from rap.utils import privacy_conversion, utility_comutation
from rap.utils.timer import Timer
from rap.workload_manager import WorkloadManager


@dataclass
class ResultManager:
    """Manages computation and storage of all mechanisms' results."""
    rap_inputs: RAPInputs
    epochs_per_eval: int
    batches_per_eval: int
    future_workload_manager: Optional[WorkloadManager]=None
    non_private_future_threshold_answers: Optional[Dict[Threshold, jnp.DeviceArray]]=None

    def __post_init__(self):
        rap_inputs = self.rap_inputs
        self.simulation_result = SimulationResult(
            rap_inputs.data_manager.dataset_name,
            rap_inputs.workload_manager,
            self.future_workload_manager,
            rap_inputs.synthetic_dataset_size,
            rap_inputs.T,
            rap_inputs.K,
            rap_inputs.epsilon,
            rap_inputs.delta,
            rap_inputs.random_key,
            rap_inputs.optimizer_hyperparams,
        )

        # Generate/log present & future errors of All-0 and GM baseline mechanisms
        self.simulation_result.all_0_present_error = self._generate_all_0_present_error()
        self.simulation_result.gm_present_error = self._generate_gm_present_error()
        if self.future_workload_manager:
            self.simulation_result.all_0_future_error = self._generate_all_0_future_error()

        # Declare field to later attach an RPResultManager to
        self.rp_result_manager = None

        # Begin timer
        self.timer = Timer()

    def _generate_all_0_present_error(self) -> float:
        """Computes the present error for the mechanism which outputs 0 for all queries.

        Returns:
            float: Max absolute error of the All-0 mechanism over all prespecified thresholds.
        """
        non_private_threshold_answers = self.rap_inputs.non_private_threshold_answers
        all_0_historical_threshold_answers = {
            threshold: 0
            for threshold in non_private_threshold_answers
        }
        return utility_comutation.compute_present_error(
            non_private_threshold_answers,
            all_0_historical_threshold_answers
        )

    def _generate_gm_present_error(self) -> float:
        """Computes the present error for the Gaussian mechanism.

        Returns:
            float: Max absolute error of the All-0 mechanism over all prespecified thresholds.
        """
        # Use a fixed random key for GM evaluation
        random_key = random.PRNGKey(42)

        # Compute GM answers for each threshold
        rap_inputs = self.rap_inputs
        rho = privacy_conversion.epsilon_delta_dp_to_rho_zcdp(rap_inputs.epsilon, rap_inputs.delta)
        gm_threshold_answers = gaussian.evaluate_on_thresholds(
            rap_inputs.data_manager.num_rows,
            rap_inputs.non_private_threshold_answers,
            rho,
            random_key
        )

        # Compute present error over all GM threshold answers
        return utility_comutation.compute_present_error(
            rap_inputs.non_private_threshold_answers,
            gm_threshold_answers
        )

    def _generate_all_0_future_error(self) -> Tuple[float, float]:
        """Computes the future error for the mechanism which outputs 0 for all queries.

        Returns:
            float: Expected max absolute error of the All-0 mechanism over randomly
                sampled thresholds.
        """
        non_private_future_threshold_answers = self.non_private_future_threshold_answers
        all_0_future_threshold_answers = {
            threshold: 0
            for threshold in non_private_future_threshold_answers
        }
        return utility_comutation.compute_future_error(
            non_private_future_threshold_answers,
            all_0_future_threshold_answers
        )

    def attach_new_rp_result_manager(
        self
    ) -> "RPResultManager":
        """Generates a new RPResultManager, stores it for use in this object (dropping its
        reference to any previous RPResultManager), then returns it to the caller.

        Returns:
            RPResultManager: The created/stored RPResultManager.
        """
        rp_result_manager = RPResultManager(
            self.rap_inputs.workload_manager,
            self.rap_inputs.non_private_threshold_answers,
            self.epochs_per_eval,
            self.batches_per_eval,
            self.future_workload_manager,
            self.non_private_future_threshold_answers,
        )
        self.rp_result_manager = rp_result_manager
        return rp_result_manager

    def log_current_rp_result(self):
        """Finalizes the currently-attached RPResultManager, acquires the corresponding
        RPResult, then appends it to the stored list of RPResults.

        Raises:
            ValueError: Throws error if attempting to log result before an current object's
                attach_new_rp_result_manager method has been called.
        """
        if not self.rp_result_manager:
            raise ValueError("No RPResultManager instantiated. Was one attached first?")
        rp_result = self.rp_result_manager.finalize()
        self.timer.add_to_total_time(rp_result.total_time)
        self.simulation_result.rp_results.append(rp_result)

    def finalize(self):
        """Performs final actions for simulation result logging. Currently, only stops timer."""
        self.simulation_result.total_time = self.timer.get_total_time()

    def save_results(
        self,
        directory: str
    ):
        """Saves the stored results into a file.

        Args:
            directory (str): Directory to store results in.
        """
        # Name file as current timestamp
        file_name = f"{time.time_ns()}.pkl"
        # Create directory if it doesn't already exist
        os.makedirs(directory, exist_ok=True)
        # Save SimulationResult to file
        with open(f"{directory}/{file_name}", "wb") as file:
            dill.dump(self.simulation_result, file)


@dataclass
class SimulationResult:
    """Dataclass for storing all mechanisms' results to be later saved."""
    dataset_name: str
    historical_workload_manager: WorkloadManager
    future_workload_manager: Optional[WorkloadManager]
    synthetic_dataset_size: int
    T: int
    K: int
    epsilon: float
    delta: float
    random_key: jnp.DeviceArray
    optimizer_hyperparams: OptimizationHyperparameters

    def __post_init__(self):
        self.all_0_present_error = None
        self.all_0_future_error = None
        self.gm_present_error = None
        self.total_time = None
        self.rp_results = []


@dataclass
class RPResultManager:
    """Computes and stores RP results through optimization."""
    historical_workload_manager: WorkloadManager
    non_private_historical_threshold_answers: Dict[Threshold, jnp.DeviceArray]
    epochs_per_eval: int
    batches_per_eval: int
    future_workload_manager: Optional[WorkloadManager]
    non_private_future_threshold_answers: Optional[Dict[Threshold, jnp.DeviceArray]]

    def __post_init__(self):
        self.rp_result = RPResult(
            self.historical_workload_manager,
            self.epochs_per_eval,
            self.batches_per_eval,
            self.future_workload_manager,
        )
        self.timer = Timer()

    def _compute_present_error(
        self,
        synthetic_dataset: jnp.DeviceArray
    ) -> float:
        """Computes the present error (max absolute error over prespecified thresholds).

        Args:
            synthetic_dataset (jnp.DeviceArray): Synthetic dataset to compute answers.

        Returns:
            float: Max error over all consistent queries of all prespecified thresholds.
        """
        all_max_errors = []
        for threshold, true_answers in self.non_private_historical_threshold_answers.items():
            synthetic_answers = self.historical_workload_manager.answer_queries_of_threshold(
                threshold,
                synthetic_dataset
            )
            all_max_errors.append(jnp.max(jnp.abs(true_answers - synthetic_answers)))
        max_error = jnp.max(jnp.array(all_max_errors))
        return max_error

    def _compute_future_error(
        self,
        synthetic_dataset: jnp.DeviceArray
    ) -> float:
        """Computes estimate of the future error (expected max absolute error over thresholds).

        Args:
            synthetic_dataset (jnp.DeviceArray): Synthetic dataset to compute answers.

        Returns:
            float: Max error over all consistent queries of randomly sampled thresholds.
        """
        all_max_errors = []
        for threshold, true_answers in self.non_private_future_threshold_answers.items():
            synthetic_answers = self.future_workload_manager.answer_queries_of_threshold(
                threshold,
                synthetic_dataset
            )
            all_max_errors.append(jnp.max(jnp.abs(true_answers - synthetic_answers)))
        all_max_errors = jnp.array(all_max_errors)
        average_max_error = jnp.mean(all_max_errors)
        std_dev_max_error = jnp.std(all_max_errors, ddof=1)
        return average_max_error, std_dev_max_error

    def log_epoch_statistics(
        self,
        synthetic_dataset: jnp.DeviceArray,
        epoch_loss: float,
        epoch_num: int,
    ):
        """Stores statistics for an epoch of thresholds.

        Args:
            synthetic_dataset (jnp.DeviceArray): Synthetic dataset to use to compute statistics.
            epoch_loss (float): Already-computed loss of the entire threshold epoch.
            epoch_num (int): Epoch number to log.
        """
        # Don't count statistics-computation time towards the RP mechanism's running time
        self.timer.stop()

        # Compute epoch loss
        self.rp_result.epoch_losses.append(
            (epoch_num, epoch_loss)
        )

        # Compute/log full present error
        present_error = self._compute_present_error(synthetic_dataset)
        self.rp_result.present_errors.append((epoch_num, present_error))

        # If future_workload, compute/log full future error
        if self.future_workload_manager:
            future_error, std_dev = self._compute_future_error(synthetic_dataset)
            self.rp_result.future_errors.append((epoch_num, future_error, std_dev))

        # Resume timing of RP mechansim
        self.timer.start()

    def log_batch_statistics(
        self,
        batch_loss: jnp.DeviceArray,
        epoch_num: int,
        batch_num: int,
    ):
        """Stores statistics for a batch of queries.

        Args:
            batch_loss (jnp.DeviceArray): Already-computed loss of a single query batch.
            epoch_num (int): Epoch number to log.
            batch_num (int): Batch number to log.
        """
        self.rp_result.batch_losses.append((epoch_num, batch_num, batch_loss))

    def finalize(self) -> "RPResult":
        """Returns the RPResult being managed, stopping the timer to compute total runtime.

        Returns:
            RPResult: Result of the RP mechanism assesed throughout its execution.
        """
        self.timer.stop()
        self.rp_result.total_time = self.timer.get_total_time()
        return self.rp_result


@dataclass
class RPResult:
    """Dataclass for storing RP result to be later saved."""
    historical_workload_manager: WorkloadManager
    epochs_per_eval: int
    batches_per_eval: int
    future_workload_manager: Optional[WorkloadManager]

    def __post_init__(self):
        self.batch_losses = []
        self.epoch_losses = []
        self.present_errors = []
        self.future_errors = []
        self.total_time = -1.
