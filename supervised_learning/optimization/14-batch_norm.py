#!/usr/bin/env python3
""" Module defines the create_batch_norm_layer method """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
        prev: tf.tensor activation output of the previous layer.
        n: number of nodes in the layer to be created.
        activation: function to be used on the output.

    Returns:
        tf.Tensor: The output of the batch normalization layer.
    """

    # Initialize layer and apply output
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    x = tf.keras.layers.Dense(n, kernel_initializer=init)(prev)

    # Calculate mean and variance of the output
    mean, variance = tf.nn.moments(x, axes=[0])

    # Initialize trainable parameters beta and gamma
    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))

    # Apply batch normalization
    normalized_x = tf.nn.batch_normalization(x, mean, variance,
                                             beta, gamma, 1e-7)

    # Return activated output
    return tf.keras.layers.Activation(activation)(normalized_x)
