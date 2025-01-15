#!/usr/bin/env python3
""" This module defines the mean_cov method """
import numpy as np


def mean_cov(X):
    """ Calculates the mean and covariance of a data set

    parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
            n: number of data points
            d: number of dimensions in each data point

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean of the data set
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix
            of the data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    # Get input dimensions
    n, d = X.shape
    
    # Calculate mean along each dimension
    mean = np.mean(X, axis=0, keepdims=True)
    
    # Calculate deviations from mean
    dev = X - mean
    
    # Covariance matrix using n-1 for unbiased estimation
    cov = np.dot(dev.T, dev) / (n - 1)

    return mean, cov
