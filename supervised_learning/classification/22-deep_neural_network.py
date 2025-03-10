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
            if not isinstance(layers[layer], int) or layers[layer] <= 0:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[layer], layers[layer - 1])
            self.weights[f"W{layer}"] = he * np.sqrt(2 / (layers[layer - 1]))
            self.weights[f"b{layer}"] = np.zeros((layers[layer], 1))

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
                key: A{layer}

        Returns:
            tuple: (A, self.__cache)
            A: output of the neural network
            self.__cache: updated cache dictionary
        """
        self.__cache["A0"] = X
        A = X
        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]

            # Calculate output current layer
            z = np.matmul(W, A) + b[0]
            A = 1 / (1 + np.exp(-z))

            # Cache output current layer
            self.__cache[f"A{layer}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cross-entropy cost of the neural network

        parameters:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) output of the neural network

        Returns:
            float: the cross-entropy cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network's predictions

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels

        Returns:
            tuple: (predictions, cost)
                predictions: numpy.ndarray with shape (1, m) holds predictions
                cost: the cross-entropy cost of the network
        """
        A, _ = self.forward_prop(X)
        # Convert probabilities to binary predictions
        predictions = np.where(A >= 0.5, 1, 0)
        # Calculate cost over all examples
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one step of gradient descent on the network

        parameters:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            cache: dictionary containing all activated outputs of each layer
            alpha: learning rate

        Returns:
            None (updates the weights and biases in-place)
        """
        # Retrieve number of examples
        m = Y.shape[1]
        # Calculate the gradient of the cost fuction layer 1
        dz = cache[f"A{self.__L}"] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache[f"A{layer - 1}"]

            # Calculate gradients
            dW = 1 / m * np.matmul(dz, A_prev.T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            dA_prev = np.matmul(self.__weights[f"W{layer}"].T, dz)
            dz = dA_prev * A_prev * (1 - A_prev)

            # Update weights and biases
            self.__weights[f"W{layer}"] -= alpha * dW
            self.__weights[f"b{layer}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: numpy.ndarray with shape (1, m) containing correct labels
            iterations: number of iterations to train over
            alpha: learning rate

        Returns:
            None (updates the weights and biases in-place)
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)
