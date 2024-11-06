#!/usr/bin/env python3
""" Module defines the dense_block method """
from tensorflow import keras as K


def dense_block(x, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks

    Parameters:
        x: tensor output of previous layer
        nb_filters: integer of filters in X
        growth_rate: growth rate for dense block
        layers: integer of layers in dense block

    Returns:
        concatenated output of each layer in dense block
        integer of filters within the concatenated outputs
    """
    initializer = K.initializers.he_normal(0)

    for _ in range(layers):
        # Stage 1: bottleneck layer (1x1 convolution)
        # Reduces input dimensionality before 3x3 convolution
        batch_norm_1 = K.layers.BatchNormalization()(x)
        relu_1 = K.layers.ReLU()(batch_norm_1)
        convolution_1 = K.layers.Conv2D(filters=4*growth_rate,
                                        kernel_size=1,
                                        padding='same',
                                        kernel_initializer=initializer)(relu_1)

        # Stage 2: Main transformation (3x3 convolution)
        # Yields growth_rate and new feature maps
        batch_norm_2 = K.layers.BatchNormalization()(convolution_1)
        relu_2 = K.layers.ReLU()(batch_norm_2)
        convolution_2 = K.layers.Conv2D(filters=growth_rate,
                                        kernel_size=3,
                                        padding='same',
                                        kernel_initializer=initializer)(relu_2)

        # Stage 3: Dense connection
        # Concatenates input and new features
        x = K.layers.concatenate([x, convolution_2])
        # Update filter count
        nb_filters += growth_rate

    # Return concatenated output and updated filter count
    return x, nb_filters
