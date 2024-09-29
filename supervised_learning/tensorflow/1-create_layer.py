#!/usr/bin/env python3
""" Module defines the create_layer method """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a new layer of a neural network where
    
    Parameters:
        prev: tensor ourput of the previous layer
        n: the number of nodes in the layer to create
        activation: the activation function the layer should use

    Returns: tensor output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')
    return layer(prev)
