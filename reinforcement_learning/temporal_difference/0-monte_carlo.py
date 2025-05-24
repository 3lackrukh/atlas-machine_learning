#!/usr/bin/env python3
"""Module defines the 0-monte_carlo method"""
import numpy as np


def monte_carlo(
                env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99
                ):
    """
    Performs the Monte Carlo algorithm:

    Parameters:
        env: environment instance.
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state, returns the next action to take
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
    Returns:
        V: numpy.ndarray of shape (s,) containing the updated value function.
    """
    for episode in range(episodes):
        # Generate episode data
        state_trajectory = []
        rewards = []

        # Reset environment
        current_state, _ = env.reset()

        # Run episode following policy
        for _ in range(max_steps):
            action = policy(current_state)
            next_state, reward, done, _, _ = env.step(action)

            rewards.append(int(reward))
            state_trajectory.append(int(current_state))
            current_state = next_state

            if done:
                break

        # Convert to numpy arrays
        rewards = np.array(rewards)
        state_trajectory = np.array(state_trajectory)

        # Update value function using Monte Carlo returns
        G = 0.0
        for t in reversed(range(len(state_trajectory))):
            state = state_trajectory[t]
            reward = rewards[t]
            G = gamma * G + reward

            # Modified first-visit check (your working approach)
            if state not in state_trajectory[:episode]:
                V[state] = V[state] + (alpha * (G - V[state]))

    return V
