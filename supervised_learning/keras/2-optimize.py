#!/usr/bin/env python3
""" Module defines the optimize_model method """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up the Adam optimization algorithm for a keras model with
    categorical crossentropy loss and accuracy metrics

    Parameters:
        network: keras model to optimize
        alpha: floating point learning rate
        beta1: first Adam optimization parameter
        beta2: second Adam optimization parameter

    Returns: None
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2)
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
