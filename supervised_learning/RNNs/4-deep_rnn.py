#!/usr/bin/env python3
"""Module defines the deep_rnn method"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Parameters:
        rnn_cells: list of RNNCell instances of length l
            that will be used for the forward propagation
        X: numpy.ndarray of shape (t, m, i)
            the data to be used
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: numpy.ndarray of shape (l, m, h)
            the initial hidden state
            l: number of layers
            m: batch size
            h: dimensionality of the hidden state
    Returns:
        H: numpy.ndarray containing all of the hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    # output dims depend on the last rnn cell
    o = rnn_cells[-1].Wy.shape[1]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    for step in range(t):
        for layer in range(l):
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
            if layer == l - 1:
                Y[step] = y
    return H, Y
