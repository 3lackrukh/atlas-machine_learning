#!/usr/bin/env python3
"""Module defines the initialize method"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gausian Mixture Model

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: int, number of clusters

    Returns:
        pi: numpy.ndarray of shape (k,) containing the priors
        m: numpy.ndarray of shape (k, d) containing the centroid means
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None, None, None

    _, d = X.shape

    # Initialize priors
    pi = np.ones(k) / k

    # Initialize centroids
    m, _ = kmeans(X, k)

    # Initialize covariance matrices for each cluster
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
