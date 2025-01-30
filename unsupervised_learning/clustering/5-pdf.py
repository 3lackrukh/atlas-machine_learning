#!/usr/bin/env python3
"""Module defines the pdf method"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the data points whose
        PDF should be evaluated
        m: numpy.ndarray of shape (d,) containing the mean of the distribution
        S: numpy.ndarray of shape (d, d) containing the covariance of the
        distribution
    Returns:
        numpy.ndarray of shape (n,) containing the PDF values fr each data
        point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    # determinant fr scaling factor
    _, d = X.shape
    det = np.linalg.det(S)

    # inverse fr quadratic formula
    inv = np.linalg.inv(S)

    # normalization constant
    const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    # difference from mean
    diff = X - m

    # Calculate PDF values using Gausian formula
    res = const * np.exp(-0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1))

    # filter for minimum value
    return np.maximum(res, 1e-300)
