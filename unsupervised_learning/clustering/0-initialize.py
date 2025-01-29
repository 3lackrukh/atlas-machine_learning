#!/usr/bin/env python3
"""Module defines the initialize method"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
        X: numpy.ndarray of size (n, d) containing the dataset.
            n: int of data points
            d: int of dimensions for each data point
        k: int of clusters.

    Returns:
        numpy.ndarray: The initialized centroids for each cluster.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    centroids = np.zeros((k, d))
    for i in range(k):
        centroids[i] = X[np.random.randint(0, n)]
    return centroids
