#!/usr/bin/env python3
""" Module defines the BayesianOptimization class"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01,
                 minimize=True):
        """
        Class constructor

        Parameters:
            f: function to be optimized
            X_init: numpy.ndarray of shape (t, 1) the inputs
                already sampled with the black-box function
            Y_init: numpy.ndarray of shape (t, 1) the outputs
                of the black-box function for each input in X_init
            bounds: tuple of (min, max) representing the bounds
                of the space in which to look for the optimal point
            ac_samples: number of samples that should be analyzed
                during acquisition
            l: float length parameter for the kernel
            sigma_f: float standard deviation given to the output of the
                black-box function
            xsi: float representing the exploration-exploitation factor
                What degree of change we consider improvement
            minimize: bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location

        Returns:
            X_next: numpy.ndarray of shape (1,) representing the next
                best sample point
            EI: numpy.ndarray of shape (ac_samples,) containing the
                expected improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        # Select optimum mean (Lowest seen or highest seen)
        if self.minimize:
            y_best = np.min(self.gp.Y)
        else:
            y_best = np.max(self.gp.Y)

        # Set error state to warn on division by zero
        with np.errstate(divide='warn'):
            # Exploration adjusted improvement at all acquisition points
            gamma = y_best - mu - self.xsi

            # improvements in standard deviations
            # Add small epsion to avoid division by zero
            Z = gamma / (sigma + 1e-10)

            # Expected improvement:
            #   probability of improvement weighted by uncertainty
            ei = gamma * norm.cdf(Z) + sigma * norm.pdf(Z)
            # No expected improvement where we have already observed
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        Parameters:
            iterations: int number of iterations to perform

        Returns:
            X_opt: numpy.ndarray of shape (1,) the optimal point
            Y_opt: numpy.ndarray of shape (1,) the optimal function value
        """

        for _ in range(iterations):
            # Find next best sample point
            X_next, _ = self.acquisition()

            # If the point has been sampled, stop
            if any(np.array_equal(X_next, x_prev) for x_prev in self.gp.X):
                break

            # Otherwise, evaluate it
            Y_next = self.f(X_next)
            # Update the Gaussian Process
            self.gp.update(X_next, Y_next)

        # Find the best sample
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        return self.gp.X[idx], self.gp.Y[idx]
