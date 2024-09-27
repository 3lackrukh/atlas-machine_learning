#!/usr/bin/env python3
""" This module defines the create_placeholders method """
import tensorflow.compact.v1 as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network

        parameters:
            nx: number of input features
            classes: number of classes

        returns: x, y placeholders
            x: placeholder for input data
            y: placeholder for one-hot encoded labels
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y

