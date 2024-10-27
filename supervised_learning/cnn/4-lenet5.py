#!/usr/bin/env python3
""" Module defines the lenet5 method """
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Builds a modified version of the LeNet-5 architecture using tensorflow

    Parameters:
        x: tf.placeholder of shape (m, 28, 28, 1)
        containing input images
        y: tf.placeholder of shape (m, 10)
        containing one-hot labels

    Returns:
        tensor containing softmax activations
        training operation utilizing Adam
        tensor containing loss of the netowork
        tensor containing accuracy of the network
    """
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size=5,
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=init)(x)
    P1 = tf.layers.MaxPooling2D(pool_size=2,
                                strides=2)(C1)
    C2 = tf.layers.Conv2D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          activation=tf.nn.relu,
                          kernel_initializer=init)(P1)
    P2 = tf.layers.MaxPooling2D(pool_size=2,
                                strides=2)(C2)
    F3 = tf.layers.Dense(units=120,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(tf.layers.Flatten()(P2))
    F4 = tf.layers.Dense(units=84,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(F3)
    output = tf.layers.Dense(units=10,
                             kernel_initializer=init)(F4)
    softmax = tf.nn.softmax(output)

    loss = tf.losses.softmax_cross_entropy(y, output)

    optimization = tf.train.AdamOptimizer().minimize(loss)

    equal = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))

    return softmax, optimization, loss, accuracy
