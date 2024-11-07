#!/usr/bin/env python3
""" Module defines the densenet121 method """
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks.

    Parameters:
        growth_rate: integer growth rate for dense block
        compression: floating point compression factor for transition layer

    Returns:
        keras.Model: The DensenetNet-121 model.
    """
    # Define input and initialize weights
    inputs = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal(0)
    filters = 64
    architecture = [6, 12, 24, 16]

    # Stage 1: Initial convolution with pre-activation
    bn1 = K.layers.BatchNormalization(axis=3)(inputs)
    relu1 = K.layers.Activation('relu')(bn1)
    conv1 = K.layers.Conv2D(filters=filters,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer)(relu1)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(conv1)
    x_stage = pool1

    # Dense blocks and transition layers
    for i in range(len(architecture)):
        # Dense block
        x_stage, filters = dense_block(x_stage, filters,
                                       growth_rate, architecture[i])

        # Transition layer (except after final block)
        if i < len(architecture) - 1:
            x_stage, filters = transition_layer(x_stage, filters, compression)

    # Classification layer
    global_avg_pool = K.layers.AveragePooling2D(
                                               padding='same')(x_stage)

    # Fully connected layer
    softmax = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=initializer)(global_avg_pool)

    return K.Model(inputs=inputs, outputs=softmax)
