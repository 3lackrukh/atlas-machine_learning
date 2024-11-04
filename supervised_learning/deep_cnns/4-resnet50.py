#!/usr/bin/env python3
""" Module defines the resnet50 method """
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015).

    Returns:
        keras.Model: The ResNet-50 model.
    """
    # Define input and initialize weights
    inputs = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal(0)
    architecture = [3, 4, 6, 3]

    # First convolutional layer
    x1 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=initializer)(inputs)
    x1 = K.layers.BatchNormalization(axis=3)(x1)
    x1 = K.layers.Activation('relu')(x1)
    x1 = K.layers.MaxPool2D(pool_size=(3, 3),
                            strides=(2, 2),
                            padding='same')(x1)

    # second convolutional layer
    x2 = projection_block(x1, [64, 64, 256], s=1)
    for _ in range(1, architecture[0]):
        x2 = identity_block(x2, [64, 64, 256])

    # third convolutional layer
    x3 = projection_block(x2, [128, 128, 512], s=2)
    for _ in range(1, architecture[1]):
        x3 = identity_block(x3, [128, 128, 512])

    # fourth convolutional layer
    x4 = projection_block(x3, [256, 256, 1024], s=2)
    for _ in range(1, architecture[2]):
        x4 = identity_block(x4, [256, 256, 1024])

    # fifth convolutional layer
    x5 = projection_block(x4, [512, 512, 2048], s=2)
    for _ in range(1, architecture[3]):
        x5 = identity_block(x5, [512, 512, 2048])

    # Average pooling
    x6 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                   padding='same')(x5)

    # Fully connected layer
    x7 = K.layers.Dense(units=1000,
                        activation='softmax',
                        kernel_initializer=initializer)(x6)

    return K.Model(inputs=inputs, outputs=x7)
