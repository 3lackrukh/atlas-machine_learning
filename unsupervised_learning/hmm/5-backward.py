#!/usr/bin/env python3
"""Module defines the backward method"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model

    Parameters:
        Observation: numpy.ndarray of shape (T,)
            the index of the observation
            T: int observations
        Emission: numpy.ndarray of shape (N, M) emission matrix
            N: int states in the model
            M: int all possible observations
        Transition: 2d numpy.ndarray of shape (N, N) transition matrix
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
            starting in a particular state

    Returns:
        P, B or None, None
            P: likelihood of the observations given the model
            B: numpy.ndarray shape (N, T)
                backward path probabilities
    """
    try:
        if not isinstance(Observation, np.ndarray) or \
                len(Observation.shape) != 1:
            return None, None
        T = Observation.shape[0]
        if not isinstance(Emission, np.ndarray) or \
                len(Emission.shape) != 2:
            return None, None
        N, _ = Emission.shape
        if not isinstance(Transition, np.ndarray) or \
                Transition.shape != (N, N):
            return None, None
        if not isinstance(Initial, np.ndarray) or \
                Initial.shape != (N, 1):
            return None, None

        # Create backward matrix size states x time-steps
        # (probability for each state and time)
        B = np.zeros((N, T))

        # Last column values set to 1
        B[:, T-1] = 1

        # Recursively compute backward probabilities
        for t in range(T-2, -1, -1):
            # sum probablities going back in time
            B[:, t] = Transition.dot(B[:, t + 1] *
                                     Emission[:, Observation[t + 1]])

        # Since backward recusion only captures future probabilities
        # We must include initial probabilities and first emission
        # to get the total probability of the observation
        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
        return P, B
    except Exception:
        return None, None
