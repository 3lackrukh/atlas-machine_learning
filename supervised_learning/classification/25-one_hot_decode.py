#!/usr/bin/env python3
""" Module defines the one_hot_encode method """
import numpy as np


def one_hot_decode(encoded_array):
    """
    Decodes a one-hot encoded numpy.ndarray
    Parameters:
        encoded_array: numpy.ndarray - shape (classes, m)
        containing the one-hot encoded labels
    Returns:
        decoded_array: numpy.ndarray - shape (m,) containing
        the numeric labels
    """
    if not isinstance(encoded_array, np.ndarray):
        return None
    if len(encoded_array.shape) != 2:
        return None
    decoded_array = np.argmax(encoded_array, axis=0)
    return decoded_array
