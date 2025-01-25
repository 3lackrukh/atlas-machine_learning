#!/usr/bin/env python3
""" Module defines the pca method """
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    Parameters:
        X: numpy.ndarray of shape n, d
            n: int of data points
            d: int of dimensions in each point
                All dimensions have a mean of 0 across all data points
        ndim: int of new dimensionality
    Returns:
        numpy.ndarray of shape n, ndim
    """
    centered_X = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(centered_X)
    return np.matmul(centered_X, vh.T[:, :ndim])
