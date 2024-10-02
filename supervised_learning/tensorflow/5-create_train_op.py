#!/usr/bin/env python3
""" Module defines the create_train_op method """
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the given loss and alpha.

    Parameters:
        loss (tf.Tensor): The loss tensor.
        alpha (float): The learning rate.

    Returns:
        tf.Operation: The training operation.
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
