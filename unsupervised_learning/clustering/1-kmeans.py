#!/usr/bin/env python3
"""Module defines the kmeans method"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Infers the K-means on a dataset

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
            n: int of data points
            d: int coordinates of each data point
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
    #  C shape (k, d) number of points and their coordinates
    C = np.random.uniform(low=np.min(X, axis=0),
                          high=np.max(X, axis=0),
                          size=(k, X.shape[1]))
    clss = None
    for i in range(iterations):

        # ASSIGN EACH POINT TO NEAREST CENTROID
        #   Compute squared L2 norm of centroids
        squared_norm_C = np.sum(C ** 2, axis=1)

        # Compute dot product between X and centroids
        D = np.dot(X, C.T)
        D *= -2

        # Add squared norms to get squared distance
        D += squared_norm_C

        # Add squared norms of X broadcasting across columns
        D += np.sum(X ** 2, axis=1)[:, np.newaxis]

        # Add small epsilon to break ties consistently
        D += np.arange(k) * 1e-12

        # Assign points to nearest centroid
        clss = np.argmin(D, axis=1)

        # Copy current centroids to compare against update
        new_C = np.copy(C)

        # Update centroids
        for j in range(k):

            # randomize centroids with no point assignments
            points = X[clss == j]
            if len(points) == 0:
                new_C[j] = np.random.uniform(low=np.min(X, axis=0),
                                             high=np.max(X, axis=0),
                                             size=(1, X.shape[1]))

            # Move centroids to center of assigned points
            else:
                new_C[j] = np.mean(points, axis=0)

        # Stop when all centroids stabilize
        #   Use approximate equality to mitigate float errors
        if np.allclose(new_C, C):
            # Recalculate the final assignments
            squared_norm_C = np.sum(new_C ** 2, axis=1)
            D = np.dot(X, new_C.T)
            D *= -2
            D += squared_norm_C
            D += np.sum(X ** 2, axis=1)[:, np.newaxis]
            D += np.arange(k) * 1e-12
            clss = np.argmin(D, axis=1)
            break

        C = new_C.copy()

    return C, clss
