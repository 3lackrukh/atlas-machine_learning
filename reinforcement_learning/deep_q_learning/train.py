#!/usr/bin/env python3
"""
Module defines the training suite for the Atari Breakout DQN agent
using keras-rl2.
"""
import numpy as np
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

# Import utilities
from utils.wrappers import GymCompatibilityWrapper
from utils.models import model_template
from utils.processors import StackDimProcessor
from utils.callbacks import EpisodicTargetNetworkUpdate


def make_env(env_id, render_mode=None):
    """
    Creates a wrapped Atari environment for use with keras-rl2
 
    Parameters:
        env_id (str): The id of the environment to create.
        render_mode (str, optional): The render mode to use.

    Returns:
        The wrapped environment.
    """
    if render_mode:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)

    # Apply Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=7,                  # No-op action for up to 7 frames
        frame_skip=4,                # Skip every 4 frames 
        screen_size=84,              # Resize to 84x84
        terminal_on_life_loss=True,  # End episode on life loss
        grayscale_obs=True,          # Convert to grayscale
        grayscale_newaxis=True,      # Keep the channel dimension
        scale_obs=False,             # Do not scale observations
    )
    
    # Create processor for dimension handling
    processor = StackDimProcessor()
    
    # Make compatible with keras-rl
    env = GymCompatibilityWrapper(env, processor)
    
    return env


def train_agent(env, state_shape, n_actions, window_length=4, steps=5000000):
    """Train a DQN agent using keras-rl2."""
    # Build DQN model
    model = model_template(state_shape, n_actions)
    model.summary()
    
    # Configure agent
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    
    # Use an annealed epsilon-greedy policy for better exploration
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )
    
    # Create the DQN agent
    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,  # Will be overridden by our callback
        policy=policy,
        enable_double_dqn=True,
        processor=StackDimProcessor()
    )
    
    # Compile DQN agent
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Create episodic target update callback
    target_update_callback = EpisodicTargetNetworkUpdate(
        update_frequency=30,  # Update target network every 30 episodes
        verbose=1
    )
    
    # Train agent
    dqn.fit(
        env,
        nb_steps=steps,
        callbacks=[target_update_callback],
        visualize=False,
        verbose=2
    )
    
    return dqn


if __name__ == "__main__":
    # Create environment
    env = make_env('BreakoutNoFrameskip-v4')

    # Get state and action dims
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Set window length for frame stacking
    window_length = 4
    
    # Set the state shape to include the window length
    state_shape = (84, 84, window_length)
    
    # Train the agent
    dqn = train_agent(env, state_shape, n_actions, window_length)
    
    # Save model weights
    print("Training complete. Saving model weights...")
    dqn.save_weights('policy.h5', overwrite=True)
    
    env.close()
