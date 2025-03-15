#!/usr/bin/env python3
"""Module defines the class BidirectionalCell"""
import numpy as np


class BidirectionalCell:
    """class representing a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
            i: integer dimensionality of the data
            h: integer dimensionality of the hidden state
            o: integer dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step

        Parameters:
            h_prev: numpy.ndarray of shape (m, h)
                m: the batch size for the data
                h: the dimensionality of the hidden state
                containing the initial hidden state
            x_t: numpy.ndarray of shape (m, i)
                m: the batch size for the data
                i: the dimensionality of the data
                containing the data for the cell
        Returns:
            h_next: numpy.ndarray of shape (m, h)
            the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step

        Parameters:
            h_next: numpy.ndarray of shape (m, h)
                m: the batch size for the data
                h: the dimensionality of the hidden state
                containing the initial hidden state
            x_t: numpy.ndarray of shape (m, i)
                m: the batch size for the data
                i: the dimensionality of the data
                containing the data for the cell
        Returns:
            h_prev: numpy.ndarray of shape (m, h)
            the previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN

        Parameters:
            H: numpy.ndarray of shape (t, m, 2 * h)
                t: the number of time steps
                m: the batch size for the data
                2 * h: the dimensionality of the hidden states
                containing the concatenated hidden states
        Returns:
            Y: numpy.ndarray of shape (t, m, o)
                t: the number of time steps
                m: the batch size for the data
                o: the dimensionality of the outputs
                containing the outputs
        """
        t, m, h2 = H.shape
        o = self.by.shape[1]
        Y = np.zeros((t, m, o))
        for step in range(t):
            # apply softmax to output
            Y[step] = np.exp(np.dot(H[step], self.Wy) + self.by)
            Y[step] = Y[step] / np.sum(Y[step], axis=1, keepdims=True)

        return Y
