#!/usr/bin/env python3
""" Module defines the learning_rate_decay mehtod """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tensorflow
    using inverse time decay

    Parameters:
        alpha: floating point original learning rate
        decay_rate: floating point weight to determine the rate of alpha decay
        decay_step: integer of gradient descent passes between decay

    Returns:
        learning_rate: the learning rate decay operation
    """
    # Create a learning rate decay operation using inverse time decay
    learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        alpha,
        decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return learning_rate
