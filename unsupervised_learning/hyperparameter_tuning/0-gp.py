#!/usr/bin/env python3
"""Module defines the 0-gp method"""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor

        Parameters:
            X_init: numpy.ndarray of shape (t, 1) the inputs
                already sampled with the black-box function

            Y_init: numpy.ndarray of shape (t, 1) the outputs
                of the black-box function for each input in X_init

            l:  float length-scale (smoothing) for the kernel
                - Larger values create smoother functions
                - Smaller values allow more rapid variations
                Must be positive

            sigma_f: float signal standard deviation
                Controls the uncertainty in the observations.
                Must be positive
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
            using Radial Basis Function (RBF)

        Parameters:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            The covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        # Square distance formula: ||X1||² + ||X2||² - 2 * X1 * X2
        sqdist = (
            np.sum(X1**2, 1).reshape(-1, 1)
            + np.sum(X2**2, 1)
            - 2 * np.dot(X1, X2.T)
            )
        # RBF kernel formula: K(X1, X2) = σ² * exp(-||X1_i - X2_j||²/(2l²))
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
