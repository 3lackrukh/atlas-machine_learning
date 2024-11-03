#!/usr/bin/env python3
""" Module defines the inception_block method """
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions (2014)

    Parameters:
        A_prev: the output from the previous layer
        filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP
            F1: integer of filters in the 1x1 convolution
            F3R: integer of filters in the 1x1 convolution
                before the 3x3 convolution
            F3: integer of filters in the 3x3 convolution
            F5R: integer of filters in the 1x1 convolution
                before the 5x5 convolution
            F5: integer of filters in the 5x5 convolution
            FPP: integer of filters in the 1x1 convolution
                after the max pooling

    Returns:
        concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution
    branch1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # 1x1 convolution before 3x3 convolution
    branch2_1x1 = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # 3x3 convolution
    branch2_3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(branch2_1x1)

    # 1x1 convolution before 5x5 convolution
    branch3_1x1 = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # 5x5 convolution
    branch3_5x5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(branch3_1x1)

    # Max pooling
    branch4_mp = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    # 1x1 convolution after max pooling
    branch4_1x1 = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(branch4_mp)

    # Concatenate outputs along the channel axis
    return K.layers.concatenate([branch1,
                                 branch2_3x3,
                                 branch3_5x5,
                                 branch4_1x1
                                 ])
