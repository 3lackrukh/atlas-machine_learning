#!/usr/bin/env python3
"""Module defines the class LSTMCell"""
import numpy as np


class LSTMCell:
    """Class that represents an LSTM unit"""
    def __init__(self, i, h, o):
        """
        Class constructor
        i: integer dimensionality of the data
        h: integer dimensionality of the hidden state
        o: integer dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev: numpy.ndarray of shape (batch_size, hidden_dim)
            containing the previous hidden state
            c_prev: numpy.ndarray of shape (batch_size, hidden_dim)
            containing the previous cell state
            x_t: numpy.ndarray of shape (batch_size, input_dim)
            containing the data input for the cell
        Returns:
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        # concatenate h_prev and x_t to match Wh dimensions
        concat = np.concatenate((h_prev, x_t), axis=1)

        # calculate forget gate
        f = np.dot(concat, self.Wf) + self.bf
        # apply sigmoid activation
        f = 1 / (1 + np.exp(-f))

        # calculate update gate
        u = np.dot(concat, self.Wu) + self.bu
        # apply sigmoid activation
        u = 1 / (1 + np.exp(-u))

        # calculate intermediate cell state
        c_inter = np.tanh(np.dot(concat, self.Wc) + self.bc)

        # calculate next cell state
        c_next = f * c_prev + u * c_inter

        # calculate output gate
        o = np.dot(concat, self.Wo) + self.bo
        # apply sigmoid activation
        o = 1 / (1 + np.exp(-o))

        # calculate next hidden state
        h_next = o * np.tanh(c_next)

        # calculate output
        y = np.dot(h_next, self.Wy) + self.by
        # apply softmax to output
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, c_next, y
