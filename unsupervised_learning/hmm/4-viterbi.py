#!/usr/bin/env python3
"""Module defines the 4-viterbi method"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states
        for a hidden markov model

    Parameters:
        Observation: numpy.ndarray of shape (T,)
            the index of each observation
            T: int observations
        Emission: numpy.ndarray of shape (N, M)
            the emission matrix
            N: int states in the model
            M: int all possible observations
        Transition: 2d numpy.ndarray of shape (N, N)
            the transition matrix
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
            starting in a particular state

    Returns:
        path, P or None, None
            path: list len(T) most likely sequence of hidden states
            P: probability of obtaining the path sequence
    """
    try:
        if not isinstance(Observation, np.ndarray) or \
                len(Observation.shape) != 1:
            return None
        T = Observation.shape[0]
        if not isinstance(Emission, np.ndarray) or \
                len(Emission.shape) != 2:
            return None
        N, _ = Emission.shape
        if not isinstance(Transition, np.ndarray) or \
                Transition.shape != (N, N):
            return None
        if not isinstance(Initial, np.ndarray) or \
                Initial.shape != (N, 1):
            return None

        # Create Viterbi matrix and backpointer matrix
        V = np.zeros((N, T))
        B = np.zeros((N, T), dtype=int)

        # First column values
        # (likelihood of any state have been first and emitted observation[0])
        V[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Recursively compute max probabilities and keep track of path
        for t in range(1, T):
            for j in range(N):
                # Find maximum probability path to current state
                probs = V[:, t-1] * Transition[:, j]

                # Store which previous state maximizes probability
                B[j, t] = np.argmax(probs)
                # Store that maximum probability x emission probability
                V[j, t] = np.max(probs) * Emission[j, Observation[t]]

        # Backtrack to find best path
        path = np.zeros(T, dtype=int)
        # Start from most likely final state
        path[-1] = np.argmax(V[:, T-1])

        # Work backward
        for t in range(T-2, -1, -1):
            # look up the state most likely transition to next
            path[t] = B[path[t+1], t+1]

        # Return probability of best path and the path
        P = np.max(V[:, T-1])
        return path, P
    except Exception:
        return None
