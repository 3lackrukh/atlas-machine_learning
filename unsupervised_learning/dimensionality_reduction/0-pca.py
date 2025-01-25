#!/usr/bin/env python3
""" Module defines the pca method """
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    
    Parameters:
        X: numpy.ndarray of shape n, d
            n: int of data points
            d: int of dimensions in each point
                All dimensions have a mean of 0 across all data points
        var: float the fraction of the variance that the PCA transformation should maintain

    Returns:
        W: numpy.ndarray weight matrix of shape d, nd
        maintaining var fraction of X's original variance.
            nd: new dimensionality of transformed X
    """
    u, s, vh = np.linalg.svd(X)
    cumsum = np.cumsum(s)
    threshold = cumsum[-1] * var
    i = 0
    while cumsum[i] < threshold:
        i += 1
    return vh.T[:, :i + 1]
