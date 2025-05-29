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
    
    def epsilon_greedy(state, Q, epsilon):
        """Choose action using ε-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.randint(env.action_space.n)
        else:
            return np.argmax(Q[state])
    
    for episode in range(episodes):
        # Initialize eligibility traces for this episode
        E = np.zeros_like(Q)
        
        # Reset environment and get initial state
        state, _ = env.reset()
        
        # Choose initial action using ε-greedy policy
        action = epsilon_greedy(state, Q, epsilon)
        
        # Episode loop
        for step in range(max_steps):
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Choose next action using ε-greedy policy (even if done)
            next_action = epsilon_greedy(next_state, Q, epsilon)
            
            # Calculate TD target based on SARSA update rule
            if done:
                # Terminal state: no future Q-value
                td_target = reward
            else:
                # Non-terminal: include discounted future Q-value
                td_target = reward + gamma * Q[next_state, next_action]
            
            # Calculate TD error: δ = r + γQ(s',a') - Q(s,a)
            td_error = td_target - Q[state, action]
            
            # Update eligibility trace for current state-action pair
            E[state, action] += 1.0
            
            # Update all Q-values using eligibility traces
            Q += alpha * td_error * E
            
            # Decay eligibility traces
            E *= gamma * lambtha
            
            # Move to next state-action pair
            if done:
                break
                
            state = next_state
            action = next_action
        
        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
    
    return Q
