#!/usr/bin/env python3
""" Module defines the identity_block method """
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015)

    Parameters:
        A_prev: output of the previous layer
        filters: tuple or list containing F11, F3, F12
            F11: integer filters in the first 1x1 convolution
            F3: integer filters in the 3x3 convolution
            F12: integer filters in the second 1x1 convolution

    Returns:
        activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(0)

    # First component of the main path
    convolution_1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    batch_norm_1 = K.layers.BatchNormalization(axis=3)(convolution_1)
    relu_1 = K.layers.Activation('relu')(batch_norm_1)

    # Second component of the main path
    convolution_2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(relu_1)
    batch_norm_2 = K.layers.BatchNormalization(axis=3)(convolution_2)
    relu_2 = K.layers.Activation('relu')(batch_norm_2)

    # Third component of the main path
    convolution_3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(relu_2)
    batch_norm_3 = K.layers.BatchNormalization(axis=3)(convolution_3)

    # Add shortcut value to the main path
    add = K.layers.Add()([batch_norm_3, A_prev])

    # Apply ReLU activation to the output
    activated_output = K.layers.Activation('relu')(add)

    return activated_output
