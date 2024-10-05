#!/usr/bin/env python3
""" Module defines the normalization_constants method """
import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """
    Calculates the normalization constants (mean and standard deviation)
    for input data X.

    Parameters:
        X (numpy.ndarray): Input data of shape (m, nx)
            m: number of data points
            nx: number of features.

    Returns:
        tuple: containing the mean and standard deviation
               of the input data.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
