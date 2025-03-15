#!/usr/bin/env python3
"""Module defines the class RNNCell"""
import numpy as np


class RNNCell:
    """
    Class RNNCell defines a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor:
            i: integer dimensionality of the data
            h: integer dimensionality of the hidden state
            o: integer dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev: numpy.ndarray of shape (batch_size, hidden_dim)
            containing the previous hidden state
            x_t: numpy.ndarray of shape (batch_size, input_dim)
            containing the data input for the cell

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        # concatenate h_prev and x_t to match Wh dimensions
        concat = np.concatenate((h_prev, x_t), axis=1)

        # calculate next hidden state
        h_next = np.tanh(np.dot(concat, self.Wh) + self.bh)

        # calculate output
        y = np.dot(h_next, self.Wy) + self.by

        # apply softmax to output
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y