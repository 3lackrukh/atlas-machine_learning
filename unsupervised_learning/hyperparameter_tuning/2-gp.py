#!/usr/bin/env python3
""" Module defines class GaussianProcess """
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
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) \
            + np.sum(X2**2, 1) \
            - 2 * np.dot(X1, X2.T)
        # RBF kernel formula: K(X1, X2) = σ² * exp(-||X1_i - X2_j||²/(2l²))
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts mean and standard deviation of points in a Gaussian process

        Parameters:
            X_s: numpy.ndarray of shape (s, 1)
                The points at which to predict the mean and standard deviation

        Returns:
            mu: numpy.ndarray of shape (s,)
                The mean for each point in X_s
            sigma: numpy.ndarray of shape (s,)
                The standard deviation for each point in X_s
        """
        # Covariance between training points and new points
        K_s = self.kernel(self.X, X_s)
        # Covariance between new points
        K_ss = self.kernel(X_s, X_s)
        # Inverse of training covariance matrix
        K_inv = np.linalg.inv(self.K)

        # Mean prediction formula: μ = K_s^T K^(-1) y
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        # Uncertainty prediction formula: σ² = K_ss - K_s^T K^(-1) K_s
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process with new observations

        Parameters:
            X_new: numpy.ndarray of shape (1,)
                The new sample point
            Y_new: numpy.ndarray of shape (1,)
                The new sample function value

        Updates the public instance attributes X and Y
        """
        # Compute new column, covariance with other points
        K_new = self.kernel(self.X, X_new.reshape(-1, 1))

        # Efficiently Construct new kernel
        self.K = np.block([
            [self.K, K_new],
            [K_new.T, self.sigma_f**2]
        ])

        # Update X and Y
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
