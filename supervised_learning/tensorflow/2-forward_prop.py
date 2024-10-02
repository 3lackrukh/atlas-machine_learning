#!/usr/bin/env python3
""" Module defines the forward_prop method """
import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """"
    Creates the forward propagation graph for the neural network.
    
    Parameters:
        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
    
    Returns:
        The prediction of the network in tensor form
    """
    create_layer = __import__('1-create_layer').create_layer
    for i, (size, activation) in enumerate(zip(layer_sizes, activations)):
        x = create_layer(x, size, activation)
    return x
