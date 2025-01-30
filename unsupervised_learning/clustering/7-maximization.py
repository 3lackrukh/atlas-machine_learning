#!/usr/bin/env python3
"""Module defines the maximization method"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm fr a GMM

    Parameters:
        X: numpy.ndarray of shape (n, d) the data points
        g: numpy.ndarray of shape (k, n) the posterior probabilities
           fr each data point in each cluster
    Returns:
        pi: numpy.ndarray of shape (k,) the updated priors fr each cluster
        m: numpy.ndarray of shape (k, d) the updated centroid means
           fr each cluster
        S: numpy.ndarray of shape (k, d, d) the updated covariance matrices
           fr each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # Calculate new priors
    pi = np.sum(g, axis=1) / n

    # Calculate new means
    g_sum = np.sum(g, axis=1)
    m = np.dot(g, X) / g_sum[:, np.newaxis]

    # Calculate new covariance matrices
    # Expand dimensions and subtract mean
    diff = X - m[:, np.newaxis, :]

    # Weight differences by posterior probabilities
    g_3D = g[:, :, np.newaxis]

    # Weighted covariance matrices for each cluster
    S = np.matmul(g_3D * diff, diff.transpose(0, 2, 1)
                  / g_sum[:, np.newaxis, np.newaxis])

    return pi, m, S
