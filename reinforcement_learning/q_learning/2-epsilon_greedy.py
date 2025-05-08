#!/usr/bin/env python3
"""Module defines the 2-epsilon_greedy method"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Parameters:
        Q: numpy.ndarray, The Q-table
        state: int, The current state
        epsilon: float, The epsilon value for exploration vs. exploitation

    Returns: int, The next action index
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(Q.shape[1])
    return action
