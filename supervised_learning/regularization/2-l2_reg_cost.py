#!/usr/bin/env python3
""" Module defines the l2_reg_cost method """
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters:
        cost: tensor containing the cost of the network
        without L2 regularization
        model: keras model including 2 layers with L2 regularization

    Returns:
        tensor containing the cost for each layer of the network
        accounting for L2 regularization
    """
    # Start with the original cost
    l2_costs = []

    for lay in model.layers:
        if isinstance(lay, tf.keras.layers.Dense) and lay.kernel_regularizer:
            # Calculate L2 regularization for this layer
            l2_cost = lay.kernel_regularizer(lay.kernel)
            # Add the L2 regularization loss to the original cost
            l2_costs.append(l2_cost)

    return cost + tf.stack(l2_costs)
