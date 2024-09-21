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
                __L, __cache, and __weights
        """
        # Validate input
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        # Architecture length
        self.__L = len(layers)

        # Set the input layer size from nx
        layers.insert(0, nx)

        # Initialize memory dictionaries
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases in each layer
        for layer in range(1, self.__L + 1):
            if layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[layer], layers[layer - 1])
            self.__weights[f"W{layer}"] = he * np.sqrt(2.0 / (layers[layer - 1]))
            self.__weights[f"b{layer}"] = np.zeros((layers[layer], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation through the network

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
            nx: number of input features
            m: number of examples

        updates:
            __cache (dict): the activated outputs of each layer
                key: A0 where input values are stored
                key: A{layer} where {layer} is the hidden layer output belongs to

        Returns:
            output (numpy.ndarray): output of the network
            cache (dict): intermediate values of the network
        """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            W = self.__weights[f"W{layer + 1}"]
            b = self.__weights[f"b{layer + 1}"]

            # Calculate ourput current layer
            z = np.matmul(W, self.__cache[f"A{layer}"]) + b
            A = 1 / (1 + np.exp(-z))

            # Cache output current layer
            self.__cache[f"A{layer + 1}"] = A
        return A, self.__cache
            