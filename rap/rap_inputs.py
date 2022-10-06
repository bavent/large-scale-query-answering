"""Data class for RAP's input parameters."""

from dataclasses import dataclass
from typing import Dict
import jax.numpy as jnp

from data.data_manager import DataManager
from rap.optimization_hyperparameters import OptimizationHyperparameters
from rap.threshold import Threshold
from rap.workload_manager import WorkloadManager


@dataclass
class RAPInputs:
    """Input parameters for the RAP mechanism."""
    data_manager: DataManager
    workload_manager: WorkloadManager
    non_private_threshold_answers: Dict[Threshold, jnp.DeviceArray]
    synthetic_dataset_size: int
    T: int
    K: int
    epsilon: float
    delta: float
    random_key: jnp.DeviceArray
    optimizer_hyperparams: OptimizationHyperparameters=OptimizationHyperparameters()
