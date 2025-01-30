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
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
