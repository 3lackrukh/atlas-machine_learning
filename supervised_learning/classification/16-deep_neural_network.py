#!/usr/bin/env python3
""" Module defines the Class DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ 
    Class defines a deep neural network
        performing binary classification
    
        Properties:
            L (int): number of network layers
            cache (dict): intermediate values of the network
            weights (dict): all weights and biases of the network
    """

    def __init__(self, nx, layers):
        """
        Class Constructor
            Inputs:
                nx (int): number of input features
                layers (list): number of nodes in each layer of the network
            
            Sets: instance properties
                L, cache, and weights
        """
        # Validate input
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        # Architecture length
        self.L = len(layers)

        # Set the input layer size from nx
        layers.insert(0, nx)

        #Initialize memory dictionaries
        self.cache = {}
        self.weights = {}

        # Initialize weights and biases in each layer
        for layer in range(1, self.L + 1):
            if layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[layer], layers[layer - 1])
            self.weights[f"W{layer}"] = he * np.sqrt(2.0 / (layers[layer - 1]))
            self.weights[f"b{layer}"] = np.zeros((layers[layer], 1))
