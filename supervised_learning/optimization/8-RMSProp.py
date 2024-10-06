#!/usr/bin/env python3
""" Module defines the create_RMSProp_op method"""
import numpy as np
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.

    Parameters:
        alpha: floating point learning rate.
        beta2: floating point RMSProp weight.
        epsilon: floating point value to avoid division by zero.

    Returns:
        optimizer
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
