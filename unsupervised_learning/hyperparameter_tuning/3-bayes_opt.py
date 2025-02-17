#!/usr/bin/env python3
""" Module defines the BayesianOptimization class"""
import numpy as np
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
            minimize: bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        _min, _max = bounds
        self.X_s = np.linspace(_min, _max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
