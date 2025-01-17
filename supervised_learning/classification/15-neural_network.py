#!/usr/bin/env python3
""" Module defines Class NeuralNetwork """
import numpy as np
import matplotlib.pyplot as plt


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

        returns: The output of the network self.__A1, self.__A2
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
        Evaluates the network's predictions

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
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        # Calculate Average cost over all examples
        cost = self.cost(Y, self.__A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one step of gradient descent on the network

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
            for the input data
            A1, A2: numpy.ndarray with shape (1, m) contains activated output
            of the respective layer for each example
            alpha: step-size of descent (default: 0.05)

        updates the private attributes __W1, __W2, __b1, __b2
        """
        # Retrieve number of examples
        m = Y.shape[1]
        # Calculate the gradient of the cost function layer 1
        dz2 = A2 - Y

        # Calculate the gradient of the weight and bias layer 1
        dw2 = 1 / m * np.dot(dz2, A1.T)
        db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(self.__W2.T, dz2) * (A1 * (1 - A1))

        dw1 = 1 / m * np.dot(dz1, X.T)
        db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

        # Update the weight and bias of layers by the step-size alpha
        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the network by updating the weights and biases
        using gradient descent

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: numpy.ndarray with shape (1, m) that contains the correct labels
                for the input data
            iterations: how many times to train (default: 5000)
            alpha: step-size of descent (default: 0.05)
            verbose: print cost every step iterations (default: True)
            graph: plot cost every step iterations (default: True)
            step: how often to plot or print (default: 100)

        updates: the private attributes __W1, __b1, __A1, __W2, __b2, __A2

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
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be a positive integer and <= iterations")

        # Initialize graphing matrix for each step
        graph_matrix = [[],[]]

        # Forward propagate and gradient descent x iterations
        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            # Graph or print cost every step iterations
            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {self.cost(Y, self.__A2)}")
                if graph:
                    graph_matrix[0].append(i)
                    graph_matrix[1].append(self.cost(Y, self.__A2))
            
        # Graph requested results
        if graph:
            plt.plot(graph_matrix[0], graph_matrix[1])
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Report updated evaluation
        return self.evaluate(X, Y)
