##########################################################
#                                                        #
#   Author: Petar Tonkovikj                              #
#                                                        #
#   General Notation:                                    #
#   N - number of nodes                                  #
#   S - number of layers                                 #
#   W - number of attributes                             #
#                                                        #
#   This file contains functions needed for solving      #
#   the minimization problem. These functions define     #
#   the goal function, its derivative, as well as co-    #
#   mpute the stationary distribution of the stocha-     #
#   stic process that defines the problem.               #
#                                                        #
# ########################################################

import numpy as np
import transition_probabilities as tp
import math

#######
#
#   Defines the loss function                                   #
#                                                               #
#   Returns: The value of the loss according to one of the      #
#   two loss function options (sqaure loss and Wilcoxon-Mann-   #
#   Whitney loss).                                              #
                                                          #######

def loss(delta, loss_funct, b=1):
    if loss_funct == "square":
        return delta**2
    if loss_funct == "wmw":
        return 1/(1 + math.exp(-delta/b))

#######
#
#   Defines the loss function derivative                        #
#                                                               #
#   Returns: The derivative of the specified loss function.     #
                                                          #######


def loss_derivative(delta, loss_funct, b=1):
    if loss_funct == "square":
        return 2*delta
    if loss_funct == "wmw":
        return (loss(delta, loss_funct, b) * (1 - loss(delta, loss_funct, b))) / b

#######
#
#   Checks whether the stationary distribution has converged    #
#                                                               #
#   Returns: True if convergence has been achieved, false       #
#   otherwise                                                   #
                                                          #######

def convergence(p1, p2, epsilon = 1e-12):
    return np.amax(np.abs(p1 - p2)) <= epsilon

#######
#
#   Computes the stationary probabilities vector and its        #
#   derivative                                                  #
#                                                               #
#   Returns: The vector of stationary distribution probabili-   #
#   ties of transition probability matrix Q and the Jacobian    #
#   matrix of partial derivatives of the vector with respect    #
#   to the parameter vector 2                                   #
                                                          #######

def PageRank(Q, dQ):
    V = Q.shape[0]
    W = dQ.shape[2]
    p = np.array([np.repeat(1/V, V)], dtype=np.float64)
    dp = np.zeros((V, W), dtype=np.float64)
    t1 = 1
    converged = False
    while not converged:
        p_new = np.empty([V])
        for i in range(V):
            p_new[i] = np.dot(p[t1-1], Q[:, i])
        p = np.append(p, [p_new], axis=0)
        converged = convergence(p_new, p[t1-1])
        t1 = t1 + 1

    for k in range(W):
        t2 = 1
        converged = False
        while not (converged or t2 == 100):
            dp_new = np.empty((V))
            for i in range(V):
                dp_new[i] = np.sum(Q[:, i] * dp[:, k] + p[min(t2, t1)-1]*dQ[:, i, k])
            converged = convergence(dp[:,k], dp_new)
            t2 = t2 + 1
            dp[:,k] = dp_new

    return p[-1], dp

#######
#
#   Defines the goal function that needs to be minimized        #
#                                                               #
#   Returns: the value of the goal function, as described in    #
#   the supervised random walks algorithm                       #
                                                          #######

def goal_function_single_node(w, *args):
    print('Calling goal function! w =', w)
    feature_matrix = args[0]
    alpha = args[1]
    start_node = args[2]
    pl = args[3]
    pd = args[4]
    loss_func = args[5]
    adjacency_matrix = args[6]
    layer_transitions = args[7]

    edge_strengths, edge_strength_derivatives = tp.generate_multiplex_edge_strength_matrices(w, feature_matrix, adjacency_matrix, layer_transitions)
    Q, dQ = tp.generate_transition_probability_matrices(edge_strengths, edge_strength_derivatives, alpha, start_node)
    p, dp = PageRank(Q, dQ)
    sum = 0
    for i in range(len(pl)):
        for j in range(len(pd)):
            sum = sum + loss(p[pl[i]] - p[pd[j]], loss_func)
    print(np.linalg.norm(w) + sum)
    return np.linalg.norm(w) + sum

#######
#
#   Defines the goal function derivative                        #
#                                                               #
#   Returns: a vector of partial derivatives of the goal        #
#   function needed by the minimizer                            #
                                                          #######

def goal_function_derivative_single_node(w, *args):
    print('Calling goal function derivative!, w =',w)
    feature_matrix = args[0]
    alpha = args[1]
    start_node =  args[2]
    pl = args[3]
    pd = args[4]
    loss_func = args[5]
    adjacency_matrix = args[6]
    layer_transitions = args[7]

    edge_strengths, edge_strength_derivatives = tp.generate_multiplex_edge_strength_matrices(w, feature_matrix, adjacency_matrix, layer_transitions)
    Q, dQ = tp.generate_transition_probability_matrices(edge_strengths, edge_strength_derivatives, alpha, start_node)

    p, dp = PageRank(Q, dQ)

    sum = np.zeros((len(w)))
    for i in range(len(pl)):
        for j in range(len(pd)):
            sum = sum + loss_derivative(p[pl[i]] - p[pd[j]], loss_func)*(dp[pl[i]] - dp[pd[j]])
    return 2*w + sum