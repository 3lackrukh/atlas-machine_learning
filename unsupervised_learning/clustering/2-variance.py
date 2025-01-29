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
        # Calculate the distance between each data point and nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # Find the index of the nearest centroid for each data point
        cluster_indices = np.argmin(distances, axis=1)

        # Calculate the distance between each data point and its centroid
        cluster_distances = distances[np.arange(X.shape[0]), cluster_indices]

        # Calculate the total intra-cluster variance
        var = np.sum(cluster_distances ** 2)

        return var

    except Exception:
        return None
