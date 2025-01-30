#!/usr/bin/env python3
"""Module defines the expectation method"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm fr a GMM

    Parameters:
        X: numpy.ndarray of shape (n, d) the data points
        pi: numpy.ndarray of shape (k,) the priors fr each cluster
        m: numpy.ndarray of shape (k, d) the centroid means fr each cluster
        S: numpy.ndarray of shape (k, d, d) covariance matrices fr each cluster
    Returns:
        g: numpy.ndarray of shape (k, n) the posterior probabilities
           fr each data point in each cluster
        l: numpy.ndarray of shape (k,) total log likelihood of expectation step
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if X.shape[1] != m.shape[1] or X.shape[1] != S.shape[1]:
        return None, None
    if S.shape[1] != S.shape[2]:
        return None, None
    if pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]:
        return None, None

    n, _ = X.shape
    k = pi.shape[0]

    # Calculate posterior probabilities
    probs = np.zeros((k, n))
    for i in range(k):
        probs[i] = pi[i] * pdf(X, m[i], S[i])

    # Calculate log likelihood
    L = np.sum(np.log(np.sum(probs, axis=0)))

    # Normalize posterior probabilities
    g = probs / np.sum(probs, axis=0)

    return g, L
