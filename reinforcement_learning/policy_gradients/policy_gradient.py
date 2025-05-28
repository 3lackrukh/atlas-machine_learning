#!/usr/bin/env python3
"""Module defines the policy method"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix
    
    Parameters:
        matrix: numpy.ndarray representing the current observation of the environment
        weight: numpy.ndarray matrix of random weights
        
    Returns:
        numpy.ndarray containing the policy probabilities for each action
    """
    # Calculate linear combination of state and weights
    z = np.dot(matrix, weight)
    
    # Apply softmax activation to get action probabilities
    # Softmax: exp(z_i) / sum(exp(z_j)) for all j
    exp_z = np.exp(z)
    policy_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    return policy_probs
