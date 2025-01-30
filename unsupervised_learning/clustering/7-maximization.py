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

    if not np.allclose(g.sum(axis=0), np.ones(n)):
        return None, None, None

    # Calculate new priors
    resp = g.sum(axis=1)
    if np.any(resp <= 0):
        return None, None, None

    pi = resp / n

    # Calculate new means
    m = np.dot(g, X) / resp[:, np.newaxis]

    # Calculate new covariance matrices
    S = np.zeros((k, d, d))
    for j in range(k):
        diff = X - m[j]
        weighted_diff = g[j, :, np.newaxis] * diff
        S[j] = np.dot(weighted_diff.T, diff) / resp[j]

    return pi, m, S
