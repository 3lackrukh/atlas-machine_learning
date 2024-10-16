#!/usr/bin/env python3
""" Module defines the l2_reg_create_layer method """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in tensorFlow
    that includes L2 regularization

    Parameters:
        prev: tensor containing the output of the previous layer
        n: integer of nodes the new layer should contain
        activation: activation function to be used on the layer
        lambtha: floating point L2 regularization parameter

    Returns:
        tensor containing the output of the new layer
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_regularizer=regularizer,
                                  kernel_initializer=init)(prev)

    return layer
