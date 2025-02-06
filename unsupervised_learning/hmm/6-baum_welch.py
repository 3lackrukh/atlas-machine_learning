#!/usr/bin/env python3
"""Module defines the baum_welch method"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, Iterations=1000):
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
    try:
        if not isinstance(Observations, np.ndarray) or \
                len(Observations.shape) != 1:
            return None, None
        T = Observations.shape[0]
        if not isinstance(Emission, np.ndarray) or \
                len(Emission.shape) != 2:
            return None, None
        N, M = Emission.shape
        if not isinstance(Transition, np.ndarray) or \
                Transition.shape != (N, N):
            return None, None
        if not isinstance(Initial, np.ndarray) or \
                Initial.shape != (N, 1):
            return None, None
        if not isinstance(Iterations, int) or Iterations < 1:
            return None, None
        for _ in range(Iterations):
            F = np.zeros((N, T))
            B = np.zeros((N, T))

            # Forward matrix first column values
            # (likelihood state was first and emitted observation[0])
            F[:, 0] = Initial.T * Emission[:, Observations[0]]
            # Recursively repeat the process for each time-step
            for t in range(1, T):
                F[:, t] = (F[:, t - 1].dot(Transition)) * \
                    Emission[:, Observations[t]]

            # Backward matrix last column values
            B[:, -1] = 1
            # Recursively compute backward probabilities
            for t in range(T-2, -1, -1):
                # sum probablities going back in time
                B[:, t] = Transition.dot(B[:, t + 1] *
                                         Emission[:, Observations[t + 1]])

            # State probability matrix (gamma)
            # probability for each state at each time-stepy
            # given by joint (forward and backward) probabilities
            # normalized at each time-step
            S = np.zeros((N, T))
            for t in range(T):
                denominator = np.sum(F[:, t] * B[:, t])
                S[:, t] = (F[:, t] * B[:, t]) / denominator

            # Transition probability tensor (xi)
            # probability for each transition given by
            # joint probabilities of states at each time-step
            # normalized at each time-step
            T_prob = np.zeros((N, N, T - 1))
            for t in range(T - 1):
                denominator = np.sum(
                    F[:, t].reshape(-1, 1) * Transition *
                    Emission[:, Observations[t + 1]].reshape(1, -1) *
                    B[:, t + 1])
                T_prob[:, :, t] = (
                    F[:, t].reshape(-1, 1) * Transition *
                    Emission[:, Observations[t + 1]].reshape(1, -1) *
                    B[:, t + 1]) / denominator
            # Update transition matrix
            Transition = np.sum(
                T_prob, 2) / np.sum(S[:, :-1], 1).reshape(-1, 1)

            # Update emission matrix
            for k in range(M):
                Emission[:, k] = np.sum(S[:, Observations == k], 1)
            Emission = Emission / np.sum(S, axis=1).reshape(-1, 1)
        return Transition, Emission

    except Exception as e:
        print(e)
        return None, None
