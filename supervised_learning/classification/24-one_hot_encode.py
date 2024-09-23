#!/usr/bin/env python3
""" Module defines the one_hot_encode method """
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Parameters:
        Y: numpy.ndarray - shape (m,) containing numeric class labels
        classes: int - number of classes
    Returns:
        one_hot: numpy.ndarray - shape (classes, m) containing
        the one-hot encoding of Y
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 1:
        return None
    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot