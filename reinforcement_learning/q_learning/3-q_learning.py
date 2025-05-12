#!/usr/bin/env python3
"""Module defines the train method"""
import numpy as np
import gymnasium as gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning

    Parameters:
        env: The FrozenLakeEnv instance
        Q: numpy.ndarray, Q-table
        episodes: int, total number of episodes to train over
        max_steps: int, maximum number of steps per episode
        alpha: float, learning rate
        gamma: float, discount rate
        epsilon: float, initial epsilon value for exploration vs. exploitation
        min_epsilon: float, minimum value that epsilon should decay to
        epsilon_decay: float, decay rate for updating epsilon between episodes

    Returns:
        Q: The updated Q-table
        total_rewards: list, containing the rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)

            # Adjust the reward for the terminal state
            if done and reward == 0:
                reward = -1

            # Accumulate episode reward
            ep_reward += reward

            # Update Q-table - handle terminal state
            target = reward if done else reward + gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])

            # Update state
            state = next_state
            if done:
                break
        # Record cumulative reward for each episode
        total_rewards.append(ep_reward)

        # Decay epsilon
        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode)

    return Q, total_rewards
