#!/usr/bin/env python3
""" Module defines the likelihood method """
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects.
    Parameters:
        x: int of patients that develop severe side effects
        n: int of patients observed
        P: 1D numpy.ndarray containing the various hypothetical
           probabilities of developing severe side effects
    Returns:
        1D numpy.ndarray containing the likelihood of obtaining
        the data x and n for each probability
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    n_choose_x = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )

    return n_choose_x * (P ** x) * ((1 - P) ** (n - x))
