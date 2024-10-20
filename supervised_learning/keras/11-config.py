#!/usr/bin/env python3
""" Module defines the save_config and load_config modules """
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model's configuration in JSON format

    Parameters:
        network: K.Model to save
        filename: string file path to save configuration to

    Returns: None
    """
    with open(filename, "w") as json_file:
        json_file.write(network.to_json())


def load_config(filename):
    """
    loads a model's configuration from JSON format

    Parameters:
        filename: string file path to load configuration from

    Returns: K.Model loaded from JSON file
    """
    with open(filename, "r") as json_file:
        return K.models.model_from_json(json_file.read())
