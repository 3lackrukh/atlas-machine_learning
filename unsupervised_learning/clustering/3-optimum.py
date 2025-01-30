#!/usr/bin/env python3
"""Module defines the optimum method"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Finds the optimum number of clusters by variance

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: int the minimum cluster count to evaluate
                (inclusive)
        kmax: int the maximum cluster count to evaluate
                (inclusive)
        iterations: int the maximum number of iterations
                allowed in K-means
    Returns:
        results: list of K-means outputs at each evaluated cluster count
        d_vars: list of variance differences between each cluster count and the
                initial count
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if kmax is not None:
        if not isinstance(kmax, int) or kmax <= 0:
            return None, None
        if kmin >= kmax:
            return None, None
    else:
        kmax = X.shape[0]

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        # Run K-means and store results
        C, clss = kmeans(X, k, iterations=iterations)
        results.append((C, clss))

        # Calculate and store variance
        var = variance(X, C)
        d_vars.append(variance(X, results[0][0]) - var)

    return results, d_vars
