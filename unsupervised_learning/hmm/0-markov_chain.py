#!/usr/bin/env python3
"""Module defines the 0-markov_chain method"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular state
        after t iterations

    Parameters:
        P: 2d numpy.ndarray of shape (n, n) the transition matrix
            P[i, j] the probability of transitioning from state i to state j
            n: int states in the markov chain
        s: 2d numpy.ndarray of shape (1, n)
            the probability of starting in each state
        t: int iterations the markov chain has been through

    Returns:
        2d numpy.ndarray of shape (1, n) the probability of
            being in a specific state after t iterations, or None on failure
    """
    try:
        if s.shape[1] > P.shape[0]:
            return None
        while t:
            s = np.matmul(s, P)
            t -= 1
        return s
    except Exception:
        return None
