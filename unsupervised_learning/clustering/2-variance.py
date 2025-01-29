#!/usr/bin/env python3
"""Module defines the variance method"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for the data set

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing the centroid means
           for each cluster
    Returns:
        var: the intra-cluster variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    try:
        # Assign points to nearest centroid
        D = np.min(np.sum((X[:, np.newaxis] - C) ** 2, axis=2), axis=1)

        # Sum all distances for total variance
        return np.sum(D)

    except Exception:
        return None
