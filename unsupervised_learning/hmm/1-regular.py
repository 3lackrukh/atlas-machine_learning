#!/usr/bin/env python3
"""Module defines the regular method"""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain

    Parameters:
        P: 2d numpy.ndarray of shape (n, n) the transition matrix
            P[i, j] the probability of transitioning from state i to state j
            n: int states in the markov chain

    Returns:
        1d numpy.ndarray of shape (1, n) the steady state probabilities,
            or None on failure
    """
    try:
        if not isinstance(P, np.ndarray):
            return None

        # Shape validation
        n = P.shape[0]
        if P.shape[1] != n:
            return None

        # Probability validation
        if (P >= 0).all() and np.all(np.isclose(P.sum(axis=1), 1)):
            evals, evecs = np.linalg.eig(P.T)

            # Check for multiple stable states (eigenvalues equal to 1)
            if np.sum(np.isclose(np.abs(evals), 1)) > 1:
                return None

            # Select and Normalize stable-state vector
            evec1 = evecs[:, np.argmax(evals)].reshape(1, n)
            return evec1 / np.sum(evec1)
        return None
    except Exception:
        return None
