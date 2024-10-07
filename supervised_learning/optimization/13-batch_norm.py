#!/usr/bin/env python3
import numpy as np


""" Module defines the batch_norm method """


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    using batch normalization.

    Parameters:
        Z: numpy.ndarray of shape (m, n)
            m: integer of data points
            n: integer of features
        gamma: numpy.ndarray of shape (1, n)
            containing the scales for batch normalization
        beta: numpy.ndarray of shape (1, n)
            containing the offsets for batch normalization
        epsilon: small floating point to avoid division by 0

    Returns:
        Z_norm: numpy.ndarray of shape (m, n)
            normalized output of neural network
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_norm = gamma * Z_norm + beta
    return Z_norm
