#!/usr/bin/env python3
""" Module defines the transition_layer method """
from tensorflow import keras as K


def transition_layer(x, nb_filters, compression):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks

    Parameters:
        x: tensor output of previous layer
        nb_filters: integer of filters in X
        compression: floating point compression factor for transition layer

    Returns:
        tensor output of transition layer
        integer of filters in output
    """
    initializer = K.initializers.he_normal(0)

    # Stage 1: Compress feature maps
    batch_norm = K.layers.BatchNormalization()(x)
    relu = K.layers.ReLU()(batch_norm)
    convolution = K.layers.Conv2D(filters=int(nb_filters*compression),
                                  kernel_size=1,
                                  padding='same',
                                  kernel_initializer=initializer)(relu)

    # Stage 2: Reduce dimensions by 2x with average pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2)(convolution)

    # Return reduced feature maps and updated filter count
    return avg_pool, int(nb_filters*compression)
