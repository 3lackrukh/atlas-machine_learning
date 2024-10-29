#!/usr/bin/env python3
""" Module defines the lenet5 method """
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras

    Parameters:
        X: K.input of shape (m, 28, 28, 1)
        containing the input images for the network
            m: integer of images
    Returns:
        a K.Model compiled to use Adam optimization
        with default hyperparameters and accuracy metrics
    """
    init = K.initializers.he_normal(seed=0)
    activation = K.activations.relu

    C1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                            activation=activation, kernel_initializer=init)(X)

    P1 = K.layers.MaxPooling2D(pool_size=2, strides=2)(C1)

    C2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                            activation=activation, kernel_initializer=init)(P1)

    P2 = K.layers.MaxPooling2D(pool_size=2, strides=2)(C2)

    F1 = K.layers.Dense(units=120, activation=activation,
                            kernel_initializer=init)(K.layers.Flatten()(P2))

    F2 = K.layers.Dense(units=84, activation=activation,
                            kernel_initializer=init)(F1)
    F3 = K.layers.Dense(units=10, activation=K.activations.softmax,
                            kernel_initializer=init)(F2)
    model = K.Model(inputs=X, outputs=F3)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss=K.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model
