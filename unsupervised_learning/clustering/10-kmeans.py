#!/usr/bin/env python3
"""Module defines the kmeans method"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs k-means on a dataset with sklearn

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: int number of clusters
    """
    k_means = sklearn.cluster.KMeans(k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_
    return C, clss
