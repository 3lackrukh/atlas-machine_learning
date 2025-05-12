#!/usr/bin/env python3
"""Module defines the 4-play method"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode with the trained agent

    Parameters:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray, the Q-table
        max_steps: the maximum number of steps in the episode

    Returns:
        total_rewards: list total rewards for the episode
        outputs: list of rendered board states at each step
    """
    total_rewards = []
    outputs = []

    state, _ = env.reset()
    outputs.append(env.render())
    ep_reward = 0
    done = False

    for step in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, done, _, _ = env.step(action)

        # Adjust the reward for the terminal state
        if done and reward == 0:
            reward = -1

        # Accumulate episode reward
        ep_reward += reward

        # Record the rendered output
        outputs.append(env.render())

        # Update state
        state = next_state
        if done:
            break

    total_rewards.append(ep_reward)
    return ep_reward, outputs
