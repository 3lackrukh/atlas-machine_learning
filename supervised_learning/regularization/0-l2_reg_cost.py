#!/usr/bin/env python3
""" Module defines the l2_reg_cost method """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters:
        cost: Floating point cost without regularization
        lambtha: Floating point regularization parameter
        weights: dictionary of weights and biases
        L: Integer of layers in the neural network
        m: Integer of data points

    Returns:
        l2_cost: floating point cost with L2 regularization
    """
    W_square_sum = 0
    
    # Calculate sum of squares of all weights
    for i in range(1, L + 1):
        W_square_sum += np.sum(np.square(weights['W' + str(i)]))
    
    # Add regularization to cost
    l2_cost = cost + (lambtha /(2 *m)) * W_square_sum
    return l2_cost
