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

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        parameters:
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example

        returns: The cost
        """
        # Retrieve number of examples
        m = Y.shape[1]
        # Calculate average cost over all examples (-1/m)
        # using cross-entropy loss algorithm of logistic regression
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data

        returns: The neuron's prediction and the cost
        """
        self.forward_prop(X)
        # Convert prediction to binary activation at threshold .5
        prediction = np.where(self.__A >= 0.5, 1, 0)
        # Calculate Average cost over all examples
        cost = self.cost(Y, self.__A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one step of gradient descent on the neuron

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
            of the neuron for each example
            alpha: step-size of descent (default: 0.05)

        updates the private attributes __W, __b
        """
        # Retrieve number of examples
        m = Y.shape[1]
        # Calculate the gradient of the cost function
        dz = A - Y
        # Calculate the gradient of the weight and bias
        dw = 1 / m * np.dot(dz, X.T)
        db = 1 / m * np.sum(dz)
        # Update the weight and bias by the step-size alpha
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron by updating the weights and biases
        using gradient descent

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
            iterations: how many times to train (default: 5000)
            alpha: step-size of descent (default: 0.05)

        updates the private attributes __W, __b, and __A
        returns: the evaluation of the training data after training
        """
        # Validate input parameters
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        # Forward propagate and gradient descent x iterations
        self.__A = 0
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        # Report updated evaluation
        return self.evaluate(X, Y)
