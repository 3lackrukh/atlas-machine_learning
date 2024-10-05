#!/usr/bin/env python3
""" Module defines the normalize method """


def normalize(X, m, s):
    """
    Normalizes a matrix of input data

    Parameters:
        X: a numpy.ndarray of shape (n, d)
            n: number of data points
            d: number of features
        m: a numpy.ndarray of shape(nx,)
            nx: contains the mean of all features of X
        s: a numpy.ndarray of shape(nx,)
            nx: contains the standard deviation of all features of X
    Returns:
        X_norm: a numpy.ndarray of shape (n, d)
            containing the normalized data
    """
    X_norm = (X - m) / s
    return X_norm
