##########################################################
#                                                        #
#   Author: Petar Tonkovikj                              #
#                                                        #
#   General Notation:                                    #
#   N - number of nodes                                  #
#   L - number of layers                                 #
#   W - number of attributes                             #
#                                                        #
#   This file contains functions that construct the      #
#   edge strength and transition probability matrices    #
#   according to the data extracted by the initializing  #
#   functions.                                           #
#                                                        #
# ########################################################

import numpy as np
import math

def calculate_edge_strength(feature_vector, w):
    """
    Defines the edge strength function
    :param feature_vector: vector of features
    :param w: vector of feature weights
    :return: The edge strength of a given edge with respect to its feature vector.
    """
    # return math.exp(np.dot(feature_vector, w))
    return np.dot(feature_vector, w) + 1

def calculate_edge_strength_derivative(feature_vector, w):
    """
    Defines the edge strength function derivative
    :param feature_vector: vector of features
    :param w: vector of feature weights
    :return: A vector of partial derivatives of the edge strength function with respect to the parameter vector  w
    """
    # return np.exp(feature_vector*w) * feature_vector
    return feature_vector

def generate_edge_strength_matrices(w, feature_vector_matrix, adjacency_matrix):
    """
    Generates edge strength and edge strength derivative matrices using the feature vector matrix
    :param w: vector of feature weights
    :param feature_vector_matrix: NxNxW matrix of edge_feature_vectors between nodes
    :param adjacency_matrix: NxN adjacency matrix of the network
    :return: A NxN sized matrix containing the edge strengths of node pairs and a NxNxW matrix containing the partial
    derivatives of these edge strengths with respect to w
    """
    edge_strength_matrix = np.empty([feature_vector_matrix.shape[0], feature_vector_matrix.shape[1]], dtype=np.float64)
    edge_strength_derivative_matrix = np.empty([feature_vector_matrix.shape[0], feature_vector_matrix.shape[1], len(w)],
                                               dtype=np.float64)
    for i in range(feature_vector_matrix.shape[0]):
        for j in range(feature_vector_matrix.shape[1]):
            edge_strength_matrix[i][j] = calculate_edge_strength(feature_vector_matrix[i][j], w) * adjacency_matrix[i][
                j]
            edge_strength_derivative_matrix[i][j] = calculate_edge_strength_derivative(feature_vector_matrix[i][j], w) * \
                                                    adjacency_matrix[i][j]
    return edge_strength_matrix, edge_strength_derivative_matrix

def generate_multiplex_edge_strength_matrices(w, feature_vector_matrices, adjacency_matrices, layer_transitions):
    """
    Generates edge strength and edge strength derivative matrices using the feature vector matrix for a multiplex
    network
    :param w: vector of feature weights
    :param feature_vector_matrices: NxNxL matrix of edge_feature_vectors between nodes in each layer
    :param adjacency_matrices: NxNxL adjacency matrix of the network for each layer
    :param layer_transitions: NxL matrix of transition probabilities of each node to each layer
    :return: A NLxNL sized matrix containing the edge strengths of node pairs for all layers and a NLxNLxW matrix
    containing the partial derivatives of these edge strengths with respect to w
    """
    num_nodes = feature_vector_matrices.shape[1]
    num_layers = feature_vector_matrices.shape[0]
    num_features = feature_vector_matrices.shape[3]
    edge_strength_matrices = np.empty((num_layers, num_nodes, num_nodes))
    edge_strength_derivative_matrices = np.empty((num_layers, num_nodes, num_nodes, num_features))
    for i in range(num_layers):
        edge_strength_matrices[i], edge_strength_derivative_matrices[i] = generate_edge_strength_matrices(w,
                                                                                        feature_vector_matrices[i],
                                                                                        adjacency_matrices[i])
    edge_strength_matrix = np.empty((num_nodes * num_layers, num_nodes * num_layers))
    edge_strength_derivative_matrix = np.empty((num_nodes * num_layers, num_nodes * num_layers, num_features))
    for i in range(num_nodes * num_layers):
        for j in range(num_nodes * num_layers):
            src_node = i % num_nodes
            dest_node = j % num_nodes
            dest_layer = j // num_nodes
            edge_strength_matrix[i][j] = layer_transitions[src_node][dest_layer] * \
                                         edge_strength_matrices[dest_layer][src_node][dest_node]
            edge_strength_derivative_matrix[i][j] = layer_transitions[src_node][dest_layer] * \
                                                    edge_strength_derivative_matrices[dest_layer][src_node][dest_node]
    return edge_strength_matrix, edge_strength_derivative_matrix

def generate_transition_probability_matrices(edge_strength_matrix, edge_strength_derivative_matrix, alpha, start_node):
    """
    Generates the transition probability matrix and its derivative from the edge strength matrix and its derivative
    matrix
    :param edge_strength_matrix: NxN matrix containing edge strengths between nodes
    :param edge_strength_derivative_matrix: NxNxW matrix containing the edge strength derivatives
    :param alpha: restart parameter
    :param start_node: random walk start node index
    :return: NxN matrix of transition probabilities between nodes and a NxNxW matrix of its derivatives
    """
    N = edge_strength_matrix.shape[0]
    W = edge_strength_derivative_matrix.shape[2]
    strength_row_sums = edge_strength_matrix.sum(
        axis=1)  # np.apply_along_axis(math.fsum, axis = 1, arr=edge_strength_matrix)#
    strength_derivative_row_sums = edge_strength_derivative_matrix.sum(axis=1)
    transition_probability_matrix = np.empty_like(edge_strength_matrix, dtype=np.float64)
    for i in range(N):
        if (strength_row_sums[i] != 0):
            transition_probability_matrix[i] = edge_strength_matrix[i] / strength_row_sums[i]
        else:
            transition_probability_matrix[i] = np.zeros(N)
    transition_probability_matrix = transition_probability_matrix * (1 - alpha)
    transition_probability_matrix[:, start_node] = transition_probability_matrix[:, start_node] + alpha
    dQ = np.empty([N, N, edge_strength_derivative_matrix.shape[2]], dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if strength_row_sums[i] != 0:
                dQ[i][j] = (1 - alpha) * (
                            edge_strength_derivative_matrix[i][j] * strength_row_sums[i] - edge_strength_matrix[i][j] *
                            strength_derivative_row_sums[i]) / (strength_row_sums[i] ** 2)
            else:
                dQ[i][j] = np.zeros(W)

    return transition_probability_matrix, dQ
