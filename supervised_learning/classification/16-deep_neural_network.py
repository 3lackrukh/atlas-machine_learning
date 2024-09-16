#!/usr/bin/env python3
""" Module defines the Class DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ Class defines a deep neural network performing binary classification """
    def __init__(self, nx, layers):
        # Validate input parameters
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if any(layer <= 0 for layer in layers):
                raise ValueError("layers must be a list of positive integers")

        # Initialize attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialize weights and biases for each layer
        for i in range(self.L):
            if i == 0:
                self.weights[f"W{i+1}"] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights[f"W{i+1}"] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            self.weights[f"b{i+1}"] = np.zeros((layers[i], 1))
        
        
