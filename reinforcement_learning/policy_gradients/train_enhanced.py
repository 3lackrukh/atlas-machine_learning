#!/usr/bin/env python3
"""Module defines the enhanced train method with analytics"""
import numpy as np
import time
from tqdm import tqdm
policy_gradient = __import__('policy_gradient').policy_gradient


def train_enhanced(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements full training using REINFORCE algorithm with comprehensive analytics
    
    Parameters:
        env: initial environment
        nb_episodes: int, number of episodes used for training
        alpha: float, the learning rate (default: 0.000045)
        gamma: float, the discount factor (default: 0.98)
        
    Returns:
        tuple: (scores, analytics_report)
    """
    # Initialize policy parameters θ randomly
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    
    # Analytics tracking
    scores = []
    high_score = 0
    high_score_episodes = []  # Episodes that achieved new high scores
    milestone_episodes = {}   # Episodes when milestones were first reached
    milestones = [100, 200, 300, 400, 500]
    start_time = time.time()
    
    # Progress bar setup
    pbar = tqdm(range(nb_episodes), desc="Training REINFORCE")
    
    # Initialize progress bar postfix
    pbar.set_postfix(score=0.0, best=0.0, records=0)
    
    # Training loop: REINFORCE algorithm
    for episode in pbar:
        # Reset environment for new episode
        state, _ = env.reset()
        
        # Episode trajectory storage
        states = []
        actions = []
        rewards = []
        gradients = []
        
        # Run one episode until termination
        done = False
        while not done:
            # Sample action aₜ ~ π(·|sₜ,θ) and compute ∇log π(aₜ|sₜ,θ)
            action, gradient = policy_gradient(state, weight)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store trajectory data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            gradients.append(gradient)
            
            state = next_state
        
        # Calculate episode score
        episode_score = sum(rewards)
        scores.append(episode_score)
        
        # Check for new high score
        if episode_score > high_score:
            high_score = episode_score
            high_score_episodes.append(episode)
        
        # Check for milestone achievements
        for milestone in milestones:
            if milestone not in milestone_episodes and episode_score >= milestone:
                milestone_episodes[milestone] = episode
        
        # Update progress bar
        pbar.set_postfix(score=episode_score, best=high_score, 
                        records=len(high_score_episodes))
        
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
    
    pbar.close()
    
    # Calculate final analytics
    training_time = time.time() - start_time
    final_100_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    
    # Generate analytics report
    analytics_report = generate_analytics_report(
        scores, high_score_episodes, milestone_episodes, 
        training_time, final_100_avg, nb_episodes
    )
    
    return scores, analytics_report


def generate_analytics_report(scores, high_score_episodes, milestone_episodes, 
                            training_time, final_100_avg, nb_episodes):
    """
    Generates comprehensive analytics report
    
    Parameters:
        scores: list of all episode scores
        high_score_episodes: list of episodes that achieved new records
        milestone_episodes: dict of milestone achievements
        training_time: total training time in seconds
        final_100_avg: average of final 100 episodes
        nb_episodes: total number of episodes
        
    Returns:
        formatted analytics report string
    """
    # Calculate high score frequency analysis
    if len(high_score_episodes) > 1:
        gaps = [high_score_episodes[i] - high_score_episodes[i-1] 
                for i in range(1, len(high_score_episodes))]
        avg_gap = np.mean(gaps)
        max_gap = max(gaps)
    else:
        gaps = []
        avg_gap = 0
        max_gap = 0
    
    # Learning phases analysis
    early_records = len([ep for ep in high_score_episodes if ep < nb_episodes * 0.25])
    late_records = len([ep for ep in high_score_episodes if ep > nb_episodes * 0.75])
    
    report = f"""
REINFORCE TRAINING ANALYTICS

PERFORMANCE SUMMARY
• Total Episodes: {nb_episodes:,}
• Training Time: {training_time/60:.1f} minutes ({training_time:.1f}s)
• Final Score: {scores[-1]:.1f}
• Best Score: {max(scores):.1f}
• Final 100-Episode Average: {final_100_avg:.1f}

HIGH SCORE PROGRESSION
• Total Records Set: {len(high_score_episodes)}
• Record Episodes: {str(high_score_episodes[:10])}{'...' if len(high_score_episodes) > 10 else ''}
• Average Gap Between Records: {avg_gap:.1f} episodes
• Longest Gap: {max_gap} episodes

MILESTONE ACHIEVEMENTS"""
    
    for milestone in sorted(milestone_episodes.keys()):
        episode = milestone_episodes[milestone]
        report += f"\n• Score {milestone}: Episode {episode:,} ({episode/nb_episodes*100:.1f}% through training)"
    
    report += f"""

LEARNING PATTERN ANALYSIS
• Early Records (0-25%): {early_records}
• Late Records (75-100%): {late_records}
• Learning Velocity: {'Fast early, slow late' if early_records > late_records else 'Consistent' if early_records == late_records else 'Accelerating'}

PERFORMANCE PHASES
• Episodes 0-{nb_episodes//4}: Avg {np.mean(scores[:nb_episodes//4]):.1f}
• Episodes {nb_episodes//4}-{nb_episodes//2}: Avg {np.mean(scores[nb_episodes//4:nb_episodes//2]):.1f}
• Episodes {nb_episodes//2}-{3*nb_episodes//4}: Avg {np.mean(scores[nb_episodes//2:3*nb_episodes//4]):.1f}
• Episodes {3*nb_episodes//4}-{nb_episodes}: Avg {np.mean(scores[3*nb_episodes//4:]):.1f}
"""
    
    return report 