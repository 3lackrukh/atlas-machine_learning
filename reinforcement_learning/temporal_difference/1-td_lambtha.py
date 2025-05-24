#!/usr/bin/env python3
"""
TD(λ) algorithm implementation for reinforcement learning
"""

import numpy as np


def td_lambtha(
                env, V, policy, lambtha, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm for value function estimation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state, returns the next action to take
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: the updated value estimate
    """

    for _ in range(episodes):
        # Reset environment and initialize eligibility traces
        state, _ = env.reset()
        e_traces = np.zeros_like(V)

        # Run episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Calculate TD error (no special terminal case)
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility traces (accumulating traces)
            e_traces[state] += 1

            # Update all states
            V += alpha * td_error * e_traces

            # Decay eligibility traces AFTER update
            e_traces *= gamma * lambtha

            state = next_state
            if done:
                break

    return V
