#!/usr/bin/env python3
"""Module defines the bi_rnn method"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_T):
    """
    Performs forward propagation for a bidirectional RNN

    Parameters:
        bi_cell: instance of BidirectionalCell class
        X: data to be used
            numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state in the forward direction
            numpy.ndarray of shape (m, h)
            h: dimensionality of the hidden state
        h_T: initial hidden state in the backward direction
            numpy.ndarray of shape (m, h)

    Returns:
        H: numpy.ndarray containing all of the concatenated hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    H = np.zeros((t, m, h * 2))
    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    h_next_f = h_0
    h_next_b = h_T

    for step in range(t):
        # forward direction
        h_next_f = bi_cell.forward(h_next_f, X[step])
        H_f[step] = h_next_f
    
        # backward direction (reversed indexing)
        b_step = -step - 1
        h_next_b = bi_cell.backward(h_next_b, X[b_step])
        H_b[b_step] = h_next_b

    H = np.concatenate((H_f, H_b), axis=2)

    Y = bi_cell.output(H)

    return H, Y
