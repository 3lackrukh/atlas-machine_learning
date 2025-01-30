#!/usr/bin/env python3
"""Module defines the expectation_maximization method"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization fr a GMM

    Parameters:
        X: numpy.ndarray of shape (n, d) the data points
        k: positive integer the number of clusters
        iterations: positive integer the maximum number of iterations
        tol: positive float the tolerance fr the log likelihood
        verbose: boolean to print log likelihood every 10 iterations
    Returns:
        pi: numpy.ndarray of shape (k,) the updated priors fr each cluster
        m: numpy.ndarray of shape (k, d) the updated centroid means
           fr each cluster
        S: numpy.ndarray of shape (k, d, d) the updated covariance matrices
           fr each cluster
        g: numpy.ndarray of shape (k, n) the posterior probabilities
           fr each data point in each cluster
        l: numpy.ndarray of shape (k,) total log likelihood of expectation step
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, L = expectation(X, pi, m, S)

    if verbose:
        print(f'Log Likelihood after 0 iterations: {L:.5f}')
    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        g, L_new = expectation(X, pi, m, S)
        

        if verbose and i % 10 == 0:
            print(f'Log Likelihood after {i} iterations: {L_new:.5f}')

        if i > 0 and abs(L - L_new) <= tol:
            if verbose:
                print(f'Log Likelihood after {i} iterations: {L_new:.5f}')
            return pi, m, S, g, L_new

        L = L_new
