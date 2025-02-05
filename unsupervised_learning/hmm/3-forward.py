#!/usr/bin/env python3
"""Module defines the forward method"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model

    Parameters:
        Observation: numpy.ndarray of shape (T,) that contains the index of the
            observation
            T: int observations
        Emission: numpy.ndarray of shape (N, M) containing the emission matrix
            N: int states in the model
            M: int all possible observations
        Transition: 2d numpy.ndarray of shape (N, N) containing the transition
            matrix
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
            starting in a particular state

    Returns:
        P, F or None, None
            P: likelihood of the observations given the model
            F: numpy.ndarray shape (N, T)
                forward path probabilities
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

        # Create forward matrix size states x time-steps
        # (probability for each forward state and time)
        F = np.zeros((N, T))

        # First column values
        # (likelihood of any state have been first and emitted observation[0])
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # Recursively repeat the process for each time-step
        for t in range(1, T):
            F[:, t] = (F[:, t - 1].dot(Transition)) * \
                Emission[:, Observation[t]]

        # Sum probabilities at last step
        P = np.sum(F[:, T-1])
        return P, F
    except Exception:
        return None, None
