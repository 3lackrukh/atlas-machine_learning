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
    m = np.zeros((k, d))
    for i in range(k):
        m[i] = np.sum(g[i, :, np.newaxis] * X, axis=0) / np.sum(g[i])

    # Calculate new covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot((g[i, :, np.newaxis] * diff).T, diff) / np.sum(g[i])

    return pi, m, S
