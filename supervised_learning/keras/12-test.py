#!/usr/bin/env python3
""" Module defines the test_model method """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Parameters:
        network: K.model to test.
        data: numpy.ndarray input data to test the model with.
        labels: numpy.ndarray true one-hot labels for the input data.
        verbose: boolean determines whether to print output during testing.

    Returns:
        numpy.ndarray loss and accuracy values for the test data.
    """
    return network.evaluate(data, labels, verbose=verbose)
