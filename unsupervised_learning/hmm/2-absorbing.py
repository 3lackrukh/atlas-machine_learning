#!/usr/bin/env python3
"""Module defines the 2-absorbing method"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing

    Parameters:
        P: 2d numpy.ndarray of shape (n, n) the transition matrix
            P[i, j] the probability of transitioning from state i to state j
            n: int states in the markov chain

    Returns:
        True if it is absorbing, or False on failure
    """
    try:
        if not isinstance(P, np.ndarray):
            return False

        # Shape validation
        n = P.shape[0]
        if P.shape[1] != n:
            return False

        # Probability validation
        if not (P >= 0).all() or not np.all(np.isclose(P.sum(axis=1), 1)):
            return False

        # Find non_absorbing states
        non_absorbing = np.where(np.diag(P) != 1)[0]

        # Must have at least one absorbing state
        if len(non_absorbing) == n:
            return False

        # If all states are absorbing, it's absorbing!
        if len(non_absorbing) == 0:
            return True

        # Extract Q matrix (transitions between non-absorbing states)
        Q = P[non_absorbing][:, non_absorbing]

        # Check if (I - Q) is nonsingular by checking if determinant is not 0
        Id = np.eye(len(Q))
        return np.linalg.det(Id - Q) != 0

    except Exception as e:
        return False
