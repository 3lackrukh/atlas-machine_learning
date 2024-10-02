#!/usr/bin/env python3
""" Module defines the calculate_loss method """
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the cross-entropy loss of a prediction.

    Parameters:
        y: placeholder for the correct labels of the input data
        y_pred: tensor containing the network's predictions

    Returns:
        Tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
