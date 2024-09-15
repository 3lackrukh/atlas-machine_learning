#!/usr/bin/env python3
""" Module defines Class NeuralNetwork """
import numpy as np


class NeuralNetwork:
    """
        Class defines a neural network with one hidden layer
        performing binary classification
    """
    def __init__(self, nx, nodes):
        """
            Neural Network Class constructor
            parameters:
                nx: number of input features
                nodes: number of hidden layer nodes
        """
        # Validate input parameters
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights, biases and activation outputs
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation through the network

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
            nx: number of input features
            m: number of examples

        returns: The output of the neuron self.__A
        """
        # Calculate the weighted sum layer 1
        z1 = np.dot(self.__W1, X) + self.__b1
        # Apply sigmoid activation function
        self.__A1 = 1 / (1 + np.exp(-z1))

        # Calculate the weighted sum layer 2
        # Layer 1 Activation as input
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        # Apply sigmoid activation function
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2
