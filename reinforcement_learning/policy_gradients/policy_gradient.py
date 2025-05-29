#!/usr/bin/env python3
"""Module defines the policy method"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    Parameters:
        matrix: numpy.ndarray representing the current
                observation of the environment
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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and weight matrix

    Parameters:
        state: numpy.ndarray the current observation of the environment
        weight: numpy.ndarray matrix of random weights

    Returns:
        action: int, the sampled action
        gradient: numpy.ndarray, the policy gradient
    """
    # Ensure state is 2D for policy function compatibility
    if state.ndim == 1:
        state = state.reshape(1, -1)

    # Get action probabilities from policy
    action_probs = policy(state, weight)

    # Sample action from the probability distribution
    action = np.random.choice(len(action_probs[0]), p=action_probs[0])

    # Compute gradient of log policy: ∇log(π(a|s))
    # For softmax policy: ∇log(π(a|s)) = state^T * (one_hot_action - π(s))

    # Create one-hot vector for the sampled action
    one_hot_action = np.zeros_like(action_probs[0])
    one_hot_action[action] = 1

    # Calculate gradient using outer product
    # gradient = state^T * (δ_a - π(a|s)) where δ_a is one-hot for action a
    if state.shape[0] == 1:
        gradient = np.outer(state[0], one_hot_action - action_probs[0])
    else:
        gradient = np.outer(state, one_hot_action - action_probs[0])

    return action, gradient
