#!/usr/bin/env python3
""" Module defines the correlation method """
import numpy as np


def correlation(C):
    """ Calculates a correlation matrix

    parameters:
        C: numpy.ndarray of shape (d, d) containing the covariance matrix
            d: number of dimensions

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        raise TypeError("C must be a 2D numpy.ndarray")
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")

    d = C.shape[0]
    
    # Initialize empty matrix
    corr = np.zeros((d, d))

    # Calculate correlation coefficients
    for i in range(d):
        for j in range(d):
            corr[i, j] = C[i, j] / np.sqrt(C[i, i] * C[j, j])

    return corr
