#!/usr/bin/env python3
"""Module defines the baum_welch method"""
import numpy as np


def baum_welch(Observations, Emission, Transition, Initial, Iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    
    Parameters:
        Observations: numpy.ndarray shape (T,)
            the index of the observation
            T: int observations
        Emission: numpy.ndarray of shape (N, M) emission matrix
            N: int hidden states
            M: int possible observations
        Transition: numpy.ndarray of shape (N, N) transition matrix
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
                 starting in a particular hidden state
        Iterations: int times expectation-maximization should be performed

    Returns:
        The converged Transition, Emission, or None, None
    """

