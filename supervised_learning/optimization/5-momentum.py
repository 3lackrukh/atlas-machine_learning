#!/usr/bin/env python3
""" Module defines the update_variables_momentum method """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with
    momentum optimization algorithm.

    Parameters:
        alpha: floating point learning rate.
        beta1: floating point momentum weight.
        var: numpy.ndarray containing variable to be updated.
        grad: numpy.ndarray gradient of the variable.
        v: The previous momentum of the variable.

    Returns:
        var_update: The updated variable.
        new_momentum: The new momentum of the variable.
   """
    # Calculate the new moment
    new_momentum = beta1 * v + (1 - beta1) * grad

    # Update the variable
    var_update = var - alpha * new_momentum

    return var_update, new_momentum
