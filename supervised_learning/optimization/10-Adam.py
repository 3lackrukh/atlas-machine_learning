#!/usr/bin/env python3
""" Module defines the create_Adam_op """
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the Adam optimization algorithm in TensorFlow

    Parameters:
        alpha: floating point learning rate
        beta1: floating point weight for the first moment
        beta2: floating point weight for the second moment
        epsilon: floating point value to prevent division by zero

    Returns:
        optimizer
    """
    return tf.keras.optimizers.Adam(learning_rate=alpha,
                                    beta_1=beta1, beta_2=beta2,
                                    epsilon=epsilon)
