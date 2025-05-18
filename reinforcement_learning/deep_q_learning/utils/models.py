#!/usr/bin/env python3
"""
Module containing neural network models for DQN agents.
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Input


def model_template(state_shape, n_actions):
    """
    Defines the DQN model architecture for policy and target networks.

    Parameters:
        state_shape (tuple): The shape of the input state.
        n_actions (int): The number of possible actions.

    Returns:
        The DQN model.
    """
    # Define model architecture from DQN paper
    # https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    model = Sequential()
    model.add(Input(shape=state_shape))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    return model
