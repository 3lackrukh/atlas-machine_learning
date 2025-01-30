#!/usr/bin/env python3
"""Module defines the gmm method"""
import sklearn.mixture


def gmm(X, k):
    """
    Performs a gmm on a dataset with sklearn

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: int number of clusters
    """
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
