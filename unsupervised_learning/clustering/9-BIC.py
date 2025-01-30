#!/usr/bin/env python3
"""Module defines the BIC method"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters fr a GMM using
        Bayesian Information Criterion

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: int minimum clusters to check (inclusive)
        kmax: int maximum clusters to check (inclusive)
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(kmin) is not int or kmin < 1:
        return None, None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= kmin:
        return None, None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    n, d = X.shape
    bic = []
    L = []
    pi, m, S = [], [], []

    # Calculate EM fr each number of clusters
    for k in range(kmin, kmax):
        pi_k, m_k, S_k, g_k, L_k = expectation_maximization(
            X, k, iterations, tol, verbose)

        # Store results
        pi.append(pi_k)
        m.append(m_k)
        S.append(S_k)
        L.append(L_k)

        # find number of parameters
        p = k * d + k * d * (d + 1) / 2 + k - 1

        # Calculate and store bic
        bic.append(p * np.log(n) - 2 * L_k)

    # Determine best k value
    bic = np.array(bic)
    L = np.array(L)
    best_k = np.argmin(bic)
    return best_k + 1, (pi[best_k], m[best_k], S[best_k]), \
        L, bic
