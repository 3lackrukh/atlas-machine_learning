#!/usr/bin/env python3
""" Module defines the save_weights and load_weights methods """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model's weights.

    Parameters:
        network: K.Model whose weights should be saved.
        filename: string file path where weights should be saved to.
        save_format: string format in which the weights should be saved.

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights.

    Parameters:
        network: K.Model whose weights should be loaded.
        filename: string file path where weights should be loaded from.

    Returns:
        None
    """
    network.load_weights(filename)
