#!/usr/bin/env python3
"""
SARSA(λ) algorithm implementation with epsilon_greedy helper method
for reinforcement learning
"""
import numpy as np


def epsilon_greedy_action(state, Q, epsilon, env):
    """
    Choose action using epsilon-greedy policy.

    Args:
        state: current state
        Q: Q-table of shape (states, actions)
        epsilon: exploration probability

    Returns:
        action: selected action
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(0, env.action_space.n)
    else:
        return np.argmax(Q[state, :])


def sarsa_lambtha(
                    env, Q, lambtha, episodes=5000, max_steps=100,
                    alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                    epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm for Q-value estimation.

    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q table
    """

    for episode in range(episodes):
        # Reset environment and initialize eligibility traces
        state, _ = env.reset()
        e_traces = np.zeros_like(Q)

        # Choose initial action using epsilon-greedy
        action = epsilon_greedy_action(state, Q, epsilon, env)

        # Run episode
        for _ in range(max_steps):
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Choose next action using epsilon-greedy
            next_action = epsilon_greedy_action(next_state, Q, epsilon, env)

            # Calculate TD target
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * Q[next_state, next_action]

            # Calculate TD error (SARSA update rule)
            td_error = td_target - Q[state, action]

            # Update eligibility traces (accumulating traces)
            e_traces[state, action] += 1

            # Update all Q-values
            Q += alpha * td_error * e_traces

            # Decay eligibility traces AFTER update
            e_traces *= gamma * lambtha

            # Move to next state and action
            state = next_state
            action = next_action

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
