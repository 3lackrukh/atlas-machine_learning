#!/usr/bin/env python3
"""
Module containing custom processors for observation and reward handling.
"""
import numpy as np
from rl.processors import Processor


class AdaptiveRewardScaler:
    """
    Adaptive reward scaler that adjusts based on best performance.
    Args:
        target_min: The minimum (negative) scaled reward
        target_best: The scaled reward for the best performance so far
        decay_factor: Factor to decay best_reward when resetting (0.95 means 5% decay)
        initial_best: Initial value for best_reward
    """
    def __init__(self, target_min=-1.0, target_best=1.2, decay_factor=0.95, initial_best=1.0):
        self.best_reward = initial_best
        self.target_min = target_min
        self.target_best = target_best
        self.decay_factor = decay_factor
        
    def scale_reward(self, shaped_reward):
        """
        Scale reward relative to best performance seen so far.
        """
        # Update best_reward tracker if we see a new best
        if shaped_reward > self.best_reward:
            self.best_reward = shaped_reward
            
        # Scale the reward
        if shaped_reward == 0:
            return 0
        elif shaped_reward > 0:
            # Scale positive rewards relative to best seen
            # This ensures the best reward gets target_best value
            return self.target_best * (shaped_reward / self.best_reward)
        else:
            # Scale negative rewards using fixed approach
            return self.target_min * min(shaped_reward / -1.0, 1.0)
            
    def reset_on_target_update(self):
        """
        Slightly decays the best reward to allow for scaling adjustment.
        """
        self.best_reward = max(1.0, self.best_reward * self.decay_factor)


class StackDimProcessor(Processor):
    """
    Custom processor that resolves dimension mismatches and implements reward shaping.
    Reward shaping is designed to mimic human-like motivation in games:
    - Breaking bricks is the primary objective and main source of satisfaction
    - Dying after scoring feels more disappointing than dying without scoring
    - Surviving longer builds anticipation and makes failure more consequential
    These human-like motivational signals help the agent learn faster by providing
    a richer reward landscape while maintaining the proper incentive hierarchy.
    """
    def __init__(self):
        super().__init__()
        self.episode_steps = 0
        self.episode_rewards = 0
        self._is_terminal = False
        self.reward_scaler = AdaptiveRewardScaler(
            target_min=-1.0,
            target_best=1.2,
            decay_factor=0.95,
            initial_best=1.0
        )
        
    def process_observation(self, observation):
        """ Return the observation as is """
        return observation
    
    def process_state_batch(self, batch):
        """
        Fixes dimension mismatch between environment obs and model inputs
        which may occur with SequentialMemory
        """
        # If we have a 5D tensor (batch, window_length, height, width, channel)
        if len(batch.shape) == 5:
            # Get dimensions
            batch_size, window_length, height, width, channels = batch.shape
            
            # Reshape to (batch, height, width, window_length*channels)
            # This stacks the frames along the channel dimension
            return np.reshape(batch, (batch_size, height, width, window_length * channels))

        return batch
    
    def process_reset(self, observation):
        """Reset episode tracking when environment resets"""
        self.episode_steps = 0
        self.episode_rewards = 0
        self._is_terminal = False
        return observation
        
    def process_reward(self, reward):
        """
        Shape rewards to provide meaningful learning signals between sparse game rewards.
        Modified to ensure strong negative signal for deaths without creating perverse incentives.
        """
        # Track accumulated rewards and steps
        self.episode_rewards += reward
        self.episode_steps += 1
        
        # Base reward (from breaking bricks)
        shaped_reward = reward
        
        # Terminal state detection (end of episode/life loss)
        if hasattr(self, '_is_terminal') and self._is_terminal:
            # Fixed penalty for all deaths: -0.5
            # This creates a consistent, strong negative signal that doesn't
            # penalize scoring behavior
            end_adjustment = -0.5
            
            # Optional: Small bonus for lasting longer (but still keeping net negative)
            survival_factor = min(1.0, self.episode_steps / 500)
            survival_bonus = 0.1 * survival_factor
            
            # Final adjustment is still negative but rewards survival
            end_adjustment += survival_bonus # At most reduces penalty to -0.4
            shaped_reward += end_adjustment
            
            # Reset episode tracking
            self.episode_steps = 0
            self.episode_rewards = 0
            self._is_terminal = False
            
        # Use adaptive scaling instead of clipping
        return self.reward_scaler.scale_reward(shaped_reward)
    
    def process_info(self, info):
        """Process game information to detect episode termination."""
        # Track terminal state for next reward processing
        if 'done' in info and info['done']:
            self._is_terminal = True
        return info
