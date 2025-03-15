#!/usr/bin/env python3
"""Module defines the class GRUCell"""
import numpy as np


class GRUCell:
    """
    Class GRUCell defines a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor:
            i: integer dimensionality of the data
            h: integer dimensionality of the hidden state
            o: integer dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.br = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev: numpy.ndarray of shape (m, h)
                m: batch size
                h: dimensionality of the hidden state
            containing the previous hidden state
            x_t: numpy.ndarray of shape (m, i)
                m: batch size
                i: dimensionality of the data
            containing the data input for the cell

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        # concatenate h_prev and x_t to match Wh dimensions
        concat = np.concatenate((h_prev, x_t), axis=1)

        # calculate update gate
        z = np.dot(concat, self.Wz) + self.bz
        # apply sigmoid activation
        z = 1 / (1 + np.exp(-z))

        # calculate reset gate
        r = np.dot(concat, self.Wr) + self.br
        # apply sigmoid activation
        r = 1 / (1 + np.exp(-r))

        # calculate intermediate hidden state
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_inter = np.tanh(np.dot(concat_r, self.Wh) + self.bh)

        # calculate next hidden state
        h_next = (1 - z) * h_prev + z * h_inter

        # calculate output
        y = np.dot(h_next, self.Wy) + self.by
        # apply softmax to output
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, y
