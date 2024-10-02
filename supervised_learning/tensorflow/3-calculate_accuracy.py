#!/usr/bin/env python3
""" Module defines the calculate_accuracy method """
import tenorflow.compat.v1 as tf
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Parameters:
        y: placeholder for the correct labels of the input data
        y_pred: tensor containing the network's predictions

    Returns:
        Tensor containing the decimal accuracy of the prediction
    """
    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy
