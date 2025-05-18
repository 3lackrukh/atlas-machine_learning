#!/usr/bin/env python3
"""
Module containing wrappers to make Gymnasium compatible with keras-rl2.
"""
import gymnasium as gym


class GymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to make gymnasium compatible with keras-rl while supporting reward shaping.
    This wrapper bridges the Gymnasium API with keras-rl expectations and ensures
    the processor receives terminal state information for proper reward shaping.
    """
    def __init__(self, env, processor=None):
        """
        Initialize the wrapper.
        
        Parameters:
            env (gym.Env): The environment to wrap.
            processor (Processor, optional): A processor to apply to observations,
                                             which can also receive episode 
                                             termination signals.
        """
        super().__init__(env)
        self.processor = processor
        
    def step(self, action):
        """
        Update step method to match keras-rl output and enhance reward shaping.
        Adds 'done' flag to info dict when episode terminates, allowing the
        processor to apply end-of-episode reward adjustments.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Signal episode termination to processor
        if done and self.processor is not None:
            info['done'] = True
            
        return obs, reward, done, info
        
    def render(self, mode=None):
        """
        Update render method to match keras-rl
        """
        return self.env.render()
    
    def reset(self, **kwargs):
        """
        Update reset method to match keras-rl output
        In gymnasium, reset returns (obs, info)
        keras-rl expects just obs
        """
        obs, _ = self.env.reset(**kwargs)
        return obs
