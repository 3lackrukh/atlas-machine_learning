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
        #   newaxis creates new X dimension (n, 1, d)
        #       to hold 2D distances from each centroid
        #   numpy broadcasting expands C before subtraction
        #       to the left  shape (1, k, d)
        #   New shape (n, k, d) encoding 2d distance between
        #   each point and each centroid
        #   .norm reduces 2d distances to 1d euclidian distances
        #   AKA an array of (n, k) hypotenuses
        #   .argmin selects the centroid with shortest hypotenuse
        D = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        clss = np.argmin(D, axis=1)

        # Copy current centroid coordinates to compare against update
        new_C = np.copy(C)

        # Update centroids
        for j in range(k):

            # randomize centroids with no point assignments
            if np.sum(clss == j) == 0:
                new_C[j] = np.random.uniform(low=np.min(X, axis=0),
                                             high=np.max(X, axis=0),
                                             size=(1, X.shape[1]))

            # Move centroids to center of assigned points
            else:
                new_C[j] = np.mean(X[clss == j], axis=0)

        # Stop when all centroids stabilize
        #   Use approximate equality to mitigate float errors
        if np.allclose(new_C, C):
            break
        C = new_C

    return C, clss
