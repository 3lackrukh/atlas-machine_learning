#!/usr/bin/env python3
""" Module defines Class Neuron """
import numpy as np


class Neuron:
    """ Class defines a single neuron performing binary classification """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron 

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
            nx: number of input features
            m: number of examples

        returns: The output of the neuron self.__A
        """
        # Calculate the weighted sum
        z = np.dot(self.__W, X) + self.__b

        # Apply sigmoid activation function
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
