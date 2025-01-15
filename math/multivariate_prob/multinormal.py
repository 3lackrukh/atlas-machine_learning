#!/usr/bin/env python3
""" Module defines the class MultiNormal """
import numpy as np


class MultiNormal():
    """ Class represents a Multivariate Normal distribution """

    def __init__(self, data):
        """ Constructor
        Parameters:
            data: numpy.ndarray of shape (n, d) containing the data set
                n: number of data points
                d: number of dimensions in each data point
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.data = data

        # Calculate mean along each dimension
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean

        # Calculate deviations from mean
        dev = data - mean

        # Covariance matrix using n-1 for unbiased estimation
        cov = np.dot(dev, dev.T) / (data.shape[1] - 1)
        self.cov = cov
