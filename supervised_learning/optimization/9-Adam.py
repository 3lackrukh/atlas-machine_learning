#!/usr/bin/env python3
""" Module defines the update_variables_Adam method """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Parameters:
        alpha: floating point learning rate.
        beta1: floating point weight for the first moment.
        beta2: floating point weight for the second moment.
        epsilon: floating point value to avoid division by zero.
        var: numpy.ndarray containing the variable to be updated.
        grad: numpy.ndarray containing the gradient of the variable.
        v: numpy.ndarray containing the first momentum.
        s: numpy.ndarray containing the second momentum.
        t: integer time step for bias correction

    Returns:
        the updated variable,
        the updated first momentum,
        the updated second momentum
    """
    # Calculate the first and second moments
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad ** 2

    # Calculate the bias-corrected moments
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    # Update the variable
    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
