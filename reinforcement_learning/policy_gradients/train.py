#!/usr/bin/env python3
"""Module defines the train method"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements full training using REINFORCE algorithm

    Parameters:
        env: initial environment
        nb_episodes: int, number of episodes used for training
        alpha: float, the learning rate (default: 0.000045)
        gamma: float, the discount factor (default: 0.98)
        show_result: bool, render env every 1000 episodes (default: False)

    Returns:
        list of all values of the score (sum of all rewards during one episode)
    """
    # Initialize policy parameters θ randomly
    # Shape: (state_space, action_space) = (4, 2) for CartPole
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    # Track scores for each episode
    scores = []

    # Training loop: REINFORCE algorithm
    for episode in range(nb_episodes):
        # Reset environment for new episode
        state, _ = env.reset()

        # Episode trajectory storage
        states = []
        actions = []
        rewards = []
        gradients = []

        # Check if we should render this episode
        render_episode = show_result and (episode % 1000 == 0)

        # Run one episode until termination
        done = False
        while not done:
            # Sample action aₜ ~ π(·|sₜ,θ) and compute ∇log π(aₜ|sₜ,θ)
            action, gradient = policy_gradient(state, weight)

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Render if requested for this episode
            if render_episode:
                try:
                    env.render()
                except Exception as e:
                    # Handle rendering errors gracefully (no display in WSL2)
                    if episode == 0:  # Only print once
                        print(f"Rendering not available: {e}")
                        print("Continuing training without visualization...")

            # Store trajectory data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            gradients.append(gradient)

            state = next_state

        # Calculate episode score
        episode_score = sum(rewards)
        scores.append(episode_score)

        # Print progress
        print(f"Episode: {episode} Score: {episode_score}")

        # Compute returns Gₜ = ∑ᵢ₌ₜᵀ γⁱ⁻ᵗ rᵢ for each timestep
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)

        # Policy gradient update: θ ← θ + α∇J(θ)
        # Where ∇J(θ) = ∑ₜ ∇log π(aₜ|sₜ,θ) * Gₜ
        for t in range(len(gradients)):
            # Scale gradient by return and learning rate
            weight += alpha * gradients[t] * returns[t]

    return scores
