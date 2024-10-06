#!/usr/bin/env python3
""" Module defines the update_variables_RMSProp method"""
import numpy as np
import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
        alpha: floaring point learning rate.
        beta2: floating point RMSProp weight.
        epsilon: floating point value to avoid division by zero.
        var: numpy.ndarray containing the variable to be updated.
        grad: numpy.ndarray containing the gradient of the variable.
        s: The previous second momentum of the variable.

    Returns:
        the updated variable
        the new momenutum
    """
    # Calculate the new second momentum
    s_new = beta2 * s + (1 - beta2) * grad ** 2

    # Calculate the update delta
    delta = alpha * grad / (np.sqrt(s_new) + epsilon)

    # Update the variable
    var_update = var - delta

    return var_update, s_new
