#!/usr/bin/env python3
""" Module defines the predict method """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Parameters:
        network: K.Model The neural network model to make prediction with
        data: numpy.ndarray input data to make the prediction with
        verbose: boolean determines whether output prints during prediction

    Returns:
        numpy.ndarray prediction of the model.
    """
    return network.predict(data, verbose=verbose)
