import jax
from data.data_manager import DataManager
from rap.dp_mechanisms import relaxed_adaptive_projection
from rap.optimization_hyperparameters import OptimizationHyperparameters
from rap.rap_inputs import RAPInputs
from rap.results_manager import ResultManager
from rap.workload_manager import WorkloadManager


# Load in dataset and get its corresponding onehot transcoder.
dataset = "ADULT"
data_manager = DataManager(dataset)
onehot_transcoder = data_manager.onehot_transcoder

# Define the privacy budget.
epsilon, delta = 0.1, 1/data_manager.num_rows**2

# Set random key for this experimental run
random_key = jax.random.PRNGKey(0)

# If testing the effect of filtering out large thresholds, set a limit to the total number of thresholds.
# Otherwise, set to None.
threshold_size_limit = None

# Create a manager for a workload consisting of some number of r-of-k thresholds.
# Initialize the workload manager with thresholds from one of the specified distributions.
r, k = 1, 3
num_thresholds = 4
workload_manager = WorkloadManager(r, k, num_thresholds, onehot_transcoder, threshold_size_limit)
workload_manager.initialize_random_thresholds(random_key, distribution_name="uniform")

# Compute the non-private answers to all consistent queries of every threshold.
non_private_threshold_answers = workload_manager.answer_queries_of_all_thresholds(
    data_manager.onehot_data
)

# Generate RAP inputs.
T, K = 4, 64
synthetic_dataset_size = 1000
rap_inputs = RAPInputs(
    data_manager,
    workload_manager,
    non_private_threshold_answers,
    synthetic_dataset_size,
    T,
    K,
    epsilon,
    delta,
    random_key,
    OptimizationHyperparameters()
)

# Instantiate a ResultManager to store the experimental results/statistics at the desired intervals.
epochs_per_eval = batches_per_eval = 1
result_manager = ResultManager(
    rap_inputs,
    epochs_per_eval,
    batches_per_eval
)

# Execute RAP mechansim simulation.
synthetic_dataset = relaxed_adaptive_projection.evaluate(
    result_manager,
    rap_inputs
)

# Save the experimental results/statistics to a file in a specified directory.
result_manager.save_results("example_results")
