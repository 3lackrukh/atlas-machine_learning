#!/usr/bin/env python3
"""Module defines the kmeans method"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Infers the K-means on a dataset

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
            n: int of data points
            d: int dimensions of each data point
        k: of clusters
        iterations: int maximum number of iterations

    Returns:
        C, clss: tuple
            C: numpy.ndarray of shape (k, d) containing
                the centroid means of each cluster
            clss: numpy.ndarray of shape (n,) containing the index of the
                cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize centroids
    C = np.random.uniform(low=np.min(X, axis=0),
                          high=np.max(X, axis=0),
                          size=(k, X.shape[1]))
    clss = None
    for i in range(iterations):

        # Assign each point to nearest centroid
        D = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
        clss = np.argmin(D, axis=0)
        new_C = np.copy(C)

        # Update centroids
        for j in range(k):

            # randomize centroids with no point assignments
            if len(X[j == clss]) == 0:
                new_C[j] = np.random.uniform(low=np.min(X, axis=0),
                                             high=np.max(X, axis=0),
                                             size=(1, X.shape[1]))

            # Move centroids to mean of assigned points
            else:
                new_C[j] = (X[j == clss].mean(axis=0))

        # Stop when all centroids fixed
        if (new_C == C).all():
            return C, clss
        C = new_C
    return C, clss
