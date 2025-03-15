#!/usr/bin/env python3
"""Module defines the 1-rnn method"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN

    Parameters:
        rnn_cell: instance of RNNCell class
        X: data to be used
            numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state
            numpy.ndarray of shape (m, h)
            h: dimensionality of the hidden state

    Returns:
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
