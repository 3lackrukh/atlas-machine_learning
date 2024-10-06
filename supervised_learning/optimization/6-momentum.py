#!/usr/bin/env python3
""" Module defines the create_momentum_op method """
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates the momentum optimization operation in tensorflow

    Parameters:
        alpha: floating point learning rate
        beta1: floating point momentum weight

    Returns:
        The momentum optimization operation
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
