#!/usr/bin/env python3
""" Module defines the likelihood, intersection,
    marginal, and posterior methods """
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


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining this data with
    the various hypothetical probabilities of developing severe side effects.
    Parameters:
        x: int of patients that develop severe side effects
        n: int of patients observed
        P: 1D numpy.ndarray containing the various hypothetical
           probabilities of developing severe side effects
        Pr: 1D numpy.ndarray containing the prior beliefs of P
    Returns:
        1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P
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
    if type(Pr) is not np.ndarray or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    # Now that we're validated, we can do what you should have done.
    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data.
    Parameters:
        x: int of patients that develop severe side effects
        n: int of patients observed
        P: 1D numpy.ndarray containing the various hypothetical
           probabilities of developing severe side effects
        Pr: 1D numpy.ndarray containing the prior beliefs of P
    Returns:
        float representing the marginal probability of obtaining x and n
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
    if type(Pr) is not np.ndarray or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    # Now that we're validated, we can do what you should have done.
    return np.sum(intersection(x, n, P, Pr))
