#!/usr/bin/env python3
"""Module defines the 0-load_env method"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLake environment from OpenAI Gymnasium

    Parameters:
        desc: List of lists representing the map of the environment
            or None to use random generated 8x8 map
        map_name: String, the name of the pre-made map to load
            or None to use random generated 8x8 map
        is_slippery: Boolean, if True, the environment will be slippery

    Returns: the environment
    """
    if desc is not None:
        env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                       is_slippery=is_slippery, render_mode='ansi')
    else:
        env = gym.make('FrozenLake-v1', map_name=map_name,
                       is_slippery=is_slippery, render_mode='ansi')
    return env
