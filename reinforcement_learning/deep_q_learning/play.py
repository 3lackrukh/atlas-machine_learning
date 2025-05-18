#!/usr/bin/env python3
"""
Script to visualize a trained DQN agent playing Atari Breakout.
"""
import argparse
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory

# Import functions from train.py
from train import make_env, model_template
from utils.processors import StackDimProcessor
from utils.patching import patch_dqn_for_continuous_training


def main(weights_path='policy.h5'):
    """
    Load and visualize a trained DQN agent playing Breakout.
    
    Parameters:
        weights_path (str): Path to the trained model weights file.
    """
    print(f"Loading weights from: {weights_path}")
    
    # Create environment with rendering enabled
    env = make_env('BreakoutNoFrameskip-v4', render_mode='human')
    
    # Get dimensions
    n_actions = env.action_space.n
    
    # Set window length for frame stacking
    window_length = 4
    
    # Set the state shape to include the window length
    state_shape = (84, 84, window_length)
    
    # Build DQN model with same architecture as training
    model = model_template(state_shape, n_actions)
    
    # Configure agent with GreedyQPolicy for evaluation (no exploration)
    memory = SequentialMemory(limit=10000, window_length=window_length)
    policy = GreedyQPolicy()
    
    dqn = DQNAgent(
        model=model,
        nb_actions=n_actions,
        memory=memory,
        nb_steps_warmup=100,  # Smaller warmup for testing
        target_model_update=10000,
        policy=policy,
        enable_double_dqn=True,
        processor=StackDimProcessor()  # Use the same processor as in training
    )
    
    # Compile the agent
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Patch the agent to avoid warmup resets - exactly as in the notebook
    dqn = patch_dqn_for_continuous_training(dqn)
    
    # Load trained weights
    try:
        dqn.load_weights(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
    except (IOError, ValueError) as e:
        print(f"Error loading weights: {e}")
        return
    
    # Test for 5 episodes
    dqn.test(env, nb_episodes=5, visualize=True)
    
    env.close()


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Play Atari Breakout with a trained DQN agent')
    parser.add_argument('--weights', type=str, default='policy.h5',
                      help='Path to the weights file (default: policy.h5)')
    parser.add_argument('--episodes', type=int, default=5,
                      help='Number of episodes to play (default: 5)')
    
    args = parser.parse_args()
    
    # Call main with the provided weights path
    main(args.weights)
