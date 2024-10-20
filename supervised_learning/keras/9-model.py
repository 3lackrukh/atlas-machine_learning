#!/usr/bin/env python3
""" Module defines the save_model and load_model methods """
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    Parameters:
        network: the model to save
        filename: string file path where the model should be saved

    Returns: None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model

    Parameters:
        filename: string file path from which to load the model

    Returns: the loaded model
    """
    return K.models.load_model(filename)
