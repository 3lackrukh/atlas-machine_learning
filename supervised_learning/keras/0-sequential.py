#!/usr/bin/env python3
""" Module defines the build_model method """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model with Keras

    Parameters:
        nx: integer of input features to the network
        layers: list of integers containing the number of nodes in each layer
        activations: list of activation functions for each layer
        lambtha: floating point L2 regularization parameter
        keep_prob: floating point probability that a node will be kept

    Returns:
        the keras model
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
