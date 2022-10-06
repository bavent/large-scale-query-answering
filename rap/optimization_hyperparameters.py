"""Data class for hyperparameters in RAP's gradient optimization procedure."""

from dataclasses import dataclass


@dataclass
class OptimizationHyperparameters:
    """ADAM optimizer hyperparameters with default values."""
    max_batch_size: int=25000
    max_epochs: int=5000
    base_learning_rate: float=1e-4
    convergence_tol: float=1e-9
