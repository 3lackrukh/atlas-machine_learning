#!/usr/bin/env python3
"""Module defines the agglomerative method"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with scipy

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: int distance threshold for clustering
    """
    hierarchy = scipy.cluster.hierarchy
    linkage = hierarchy.linkage(X, method='ward')
    clss = hierarchy.fcluster(linkage, dist, criterion='distance')
    hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
