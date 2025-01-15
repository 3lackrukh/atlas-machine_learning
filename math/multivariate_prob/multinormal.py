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
        
    def pdf(self, x):
        """ Calculates the PDF at a data point
        Parameters:
            x: numpy.ndarray of shape (d,) containing the data point
                d: number of dimensions of the Multivariate Normal distribution
        Returns:
            the PDF values for the data point
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2\
            or x.shape[0] != self.cov.shape[1]\
            or x.shape[1] != 1:
            raise ValueError(f"x must have the shape ({self.cov.shape[0]}, 1)")

        d = self.cov.shape[1]
        
        # Calculate determinant and inverse of covariance matrix
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        
        # Calculate deviations from mean
        dev = float(x - self.mean)
        
        # Calculate quadratic and normalization
        quad = np.dot(np.dot(dev.T, inv), dev)
        norm = 1 / np.sqrt((2 * np.pi) ** d * det)
        
        # Probability Density Function
        pdf = float(norm * np.exp(-0.5 * quad))
        return pdf
