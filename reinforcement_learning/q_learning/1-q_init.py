#!/usr/bin/env python3
"""Module defines the 1-q_init method"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    Parameters:
        env: The FrozenLakeEnv instance

    Returns: numpy.ndarray, The initialized Q-table
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    return q_table
