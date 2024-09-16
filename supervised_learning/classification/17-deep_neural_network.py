#!/usr/bin/env python3
""" Module defines the Class DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ Class defines a deep neural network with binary classification """
    def __init__(self, nx, layers):
        # Validate input parameters
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        # Initialize attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialize weights and biases in each layer
        for l in range(self.L):
            if layers[l] < 1:
                raise TypeError("layers must be a list of positive integers")
            if l == 0:
                self.weights[f"W{l+1}"] = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)
            else:
                self.weights[f"W{l+1}"] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(2 / layers[l-1])
            self.weights[f"b{l+1}"] = np.zeros((layers[l], 1))

    @property
    def L(self):
        return self.__L
    
    @property
    def cache(self):
        return self.__cache
    
    @property
    def weights(self):
        return self.__weights
