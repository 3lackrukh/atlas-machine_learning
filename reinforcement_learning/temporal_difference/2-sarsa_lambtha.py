#!/usr/bin/env python3
"""
Module defines the sarsa_lambtha method for SARSA(λ) algorithm implementation
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm for temporal difference learning.

    SARSA(λ) uses eligibility traces to update Q-values for all state-action
    pairs visited in an episode, with exponentially decaying weights.

    Args:
        env: environment instance
        Q: numpy.ndarray of shape (s,a) containing the Q table
        lambtha: eligibility trace factor (λ ∈ [0,1])
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate (α)
        gamma: discount rate (γ)
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q table
    """
    og_epsilon = epsilon

    for episode in range(episodes):
        # Reset environment and initialize eligibility traces
        state, _ = env.reset()
        e_traces = np.zeros_like(Q)

        # Choose initial action using epsilon-greedy
        if np.random.random() < epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(Q[state, :])

        # Run episode
        for step in range(max_steps):
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Choose next action using epsilon-greedy
            if np.random.random() < epsilon:
                next_action = np.random.randint(0, env.action_space.n)
            else:
                next_action = np.argmax(Q[next_state, :])

            # Calculate TD error (SARSA update rule)
            td_error = reward + gamma * Q[next_state, next_action] - \
                Q[state, action]

            # Decay eligibility traces from previous steps
            e_traces *= gamma * lambtha

            # Update eligibility traces (accumulating traces)
            e_traces[state, action] += 1

            # Update all Q-values
            Q += alpha * td_error * e_traces

            # Move to next state and action
            state = next_state
            action = next_action

            if done:
                break

        # Decay epsilon after each episode (exponential decay)
        epsilon = min_epsilon + (og_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

    return Q
