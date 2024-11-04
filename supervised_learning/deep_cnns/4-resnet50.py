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

    # First stage
    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=initializer)(inputs)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(bn1)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(relu1)

    # second stage
    stage2 = projection_block(pool1, [64, 64, 256], s=1)
    for _ in range(1, architecture[0]):
        stage2 = identity_block(stage2, [64, 64, 256])

    # third stage
    stage3 = projection_block(stage2, [128, 128, 512], s=2)
    for _ in range(1, architecture[1]):
        stage3 = identity_block(stage3, [128, 128, 512])

    # fourth stage
    stage4 = projection_block(stage3, [256, 256, 1024], s=2)
    for _ in range(1, architecture[2]):
        stage4 = identity_block(stage4, [256, 256, 1024])

    # fifth stage
    stage5 = projection_block(stage4, [512, 512, 2048], s=2)
    for _ in range(1, architecture[3]):
        stage5 = identity_block(stage5, [512, 512, 2048])

    # Average pooling
    pool6 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      padding='same')(stage5)

    # Fully connected layer
    softmax7 = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer)(pool6)

    return K.Model(inputs=inputs, outputs=softmax7)
