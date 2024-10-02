#!/usr/bin/env python3
""" Module defines the evaluate method """
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Parameters:
        X: Input data placeholder.
        Y: One-hot labels for X.
        save_path: Path to the saved model.

    Returns:
        prediction, accuracy, loss
    """
    # Load the saved model
    loaded_model = tf.keras.models.load_model(save_path)

    # Create a session and evaluate the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prediction = sess.run(loaded_model.output, feed_dict={X: X})
        accuracy = sess.run(loaded_model.accuracy, feed_dict={X: X, Y: Y})
        loss = sess.run(loaded_model.loss, feed_dict={X: X, Y: Y})

    return prediction, accuracy, loss
