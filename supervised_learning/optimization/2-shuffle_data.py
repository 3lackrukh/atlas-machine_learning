#!/usr/bin/env python3
""" Module defines the shuffle_data method """
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Parameters:
        X: (numpy.ndarray) Input data of shape (m, nx)
            m: (int) Number of data points
            nx: (int) Number of features
        Y (numpy.ndarray): Labels of shape (m, ny)
            m: (int) Number of data points
            ny: (int) Number of features

    Returns:
        X_shuffled: (numpy.ndarray) Shuffled X
        Y_shuffled: (numpy.ndarray) Shuffled Y
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
