#!/usr/bin/env python3
""" Module defines dropout_create_layer method """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network with dropout regularization.

    Parameters:
        prev: Tensor containing output of previous layer of the neural network.
        n: integer of nodes in the new layer.
        activation: activation function to be used for the new layer.
        keep_prob: floating point probability that a node will be kept.
        training: boolean indicating whether the model is in training mode or not.

    Returns:
        Tensor containing output of the new layer with dropout regularization applied.
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg')
    
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)
    dense = tf.keras.layers.Dense(units=n, 
                                  activation=activation,
                                  kernel_initializer=init)
    
    x = dense(prev)
    return dropout(x, training=training)
    