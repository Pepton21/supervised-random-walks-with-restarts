import numpy as np
from scipy import optimize
import minimization_problem as mp
import transition_probabilities as tp
import init
import time

# Set an initial valie for w
w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float)

# Perform initialization
map = init.get_node_map()
adjacency_matrix = np.load('stored_data/adjacency_matrices.npy')
edge_attributes = np.load("stored_data/feature_matrices.npy")
layer_transitions = np.load("stored_data/layer_transitions.npy")
alpha = 0.3
start_node = 694
loss_function = "wmw"
D = np.load("stored_data/positive_samples.npy")
L = np.load("stored_data/negative_samples.npy")

# Create matrix for storing optimal values for w from different starting nodes
results = np.empty([0, len(w)])

# Create the bonds for each parameter
bounds = []
for i in range(len(w)):
    bounds += [(0, None)]

# Calculate number of nodes and layers
num_layers = layer_transitions.shape[1]
num_nodes = len(map)
total_nodes = num_layers * num_nodes

# Start Measuring training time
global_start_time = time.time()
count = 0

# Perform training
for i in range(num_nodes):
    for j in range(num_layers):
        # Perform the optimization using L-BFGS
        global_start_time = time.time()
        arguments = (edge_attributes, alpha, start_node, L[start_node], D[start_node], loss_function, adjacency_matrix, layer_transitions)
        start_time = time.time()
        w_result = optimize.minimize(fun = mp.goal_function_single_node, x0=w, args = arguments, method="L-BFGS-B", jac=mp.goal_function_derivative_single_node, bounds=bounds, options={'ftol': 1e-3})
        # Print results of optimization
        print('Partial result finished in', time.time() - start_time, 'seconds!')
        print("Result:")
        print(w_result)
        # Append result to list of weight vectors
        results = np.append(results, [w_result.x], axis=0)
        # Print training progress
        count = count + 1
        print('Progress:', i * num_layers, 'out of', total_nodes)

# Print results of training
print('Finished in', time.time() - global_start_time)
print('Results:')
print(results)
w_optimal = np.sum(results, axis = 0) / total_nodes
print('Optimal w:', w_optimal)

# Save results to file
np.save("stored_data/obtained_weight_vectors", results)
np.save("stored_data/average_weight_vector", w_optimal)