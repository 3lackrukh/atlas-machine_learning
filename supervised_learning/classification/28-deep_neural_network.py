#!/usr/bin/env python3
""" Module defines the Class DeepNeuralNetwork """
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Class defines a deep neural network
        performing binary classification

        Properties:
            L (int): number of network layers
            cache (dict): intermediate values of the network
            weights (dict): all weights and biases of the network
    """

    def __init__(self, nx, layers, activation="sig"):
        """
        Class Constructor
            Inputs:
                nx (int): number of input features
                layers (list): number of nodes in each layer of the network
                activation (str): activation function to be used
                    (default: "sig" for sigmoid)

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
        if activation not in ["sig", "tanh"]:
            raise ValueError("activation must be 'sig' or 'tanh'")

        # Architecture length
        self.__L = len(layers)

        # Set the input layer size from nx
        layers.insert(0, nx)

        # Initialize memory dictionaries
        self.__cache = {}
        self.__weights = {}

        # Initialize activation function
        self.__activation = activation

        # Initialize weights and biases in each layer
        for layer in range(1, self.__L + 1):
            if layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            he = np.random.randn(layers[layer], layers[layer - 1])
            self.weights[f"W{layer}"] = he * np.sqrt(2.0 / (layers[layer - 1]))
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
    
    @property
    def activation(self):
        return self.__activation

    def softmax(self, z):
        """
        Calculates the softmax activation function

        parameters:
            z: numpy.ndarray with shape (nx, m) nx is the number of features
                and m is the number of examples

        Returns:
            numpy.ndarray: the softmax activation of z
        """
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def sig(self, z):
        """
        Calculates the sigmoid activation function

        parameters:
            z: numpy.ndarray with shape (nx, m) nx number of features
                and m is the number of examples

        Returns:
            numpy.ndarray: the sigmoid activation of z
        """
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        """
        Calculates the hyperbolic tangent activation function

        parameters:
            z: numpy.ndarray with shape (nx, m) nx number of features
                and m is the number of examples

        Returns:
            numpy.ndarray: the hyperbolic tangent activation of z
        """
        return np.tanh(z)

    @staticmethod
    def one_hot_encode(Y, classes):
        """
        Converts a numeric label vector into a one-hot matrix
        Parameters:
            Y: numpy.ndarray - shape (m,) containing numeric class labels
            classes: int - number of classes
        Returns:
            one_hot: numpy.ndarray - shape (classes, m) containing
            the one-hot encoding of Y
        """
        if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
            return None
        if not isinstance(classes, int) or classes < 2 or classes < Y[max(Y-1)]:
            return None

        try:
            one_hot = np.zeros((classes, Y.shape[0]))
            one_hot[Y, np.arange(Y.shape[0])] = 1
            return one_hot
        except IndexError:
            return None

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
            if layer == self.__L:
                # Softmax activation for output layer
                A = self.softmax(z)
            else:
                # Activation for hidden layers
                A = getattr(self, self.__activation)(z)

            # Cache output current layer
            self.__cache[f"A{layer}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cross-entropy cost of the neural network

        parameters:
            Y: a one-hot numpy.ndarray with shape (classes, m)
            A: numpy.ndarray with shape (classes, m) neural network output

        Returns:
            float: the cross-entropy cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network's predictions

        parameters:
            X: numpy.ndarray with shape (nx, m) that contains the input data
                nx: number of input features
                m: number of examples
            Y: one-hot numpy.ndarray with shape (classes, m) expected output

        Returns:
            tuple: (activations, cost)
                activations: one-hot numpy.ndarray with shape (classes, m)
                cost: the cross-entropy cost of the network
        """
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        activations = self.one_hot_encode(predictions, Y.shape[0])
        cost = self.cost(Y, A)
        return activations, cost

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

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
            # Initialize graphing matrix for each step
            graph_matrix = [[], []]

        # Forward propagate and gradient descent x iterations
        for i in range(iterations + 1):
            A, self.__cache = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)

            # Print and or plot cost every step iterations
            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {self.cost(Y, A)}")
                if graph:
                    graph_matrix[0].append(i)
                    graph_matrix[1].append(self.cost(Y, A))

        # Graph requested results
        if graph:
            plt.plot(graph_matrix[0], graph_matrix[1])
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        # Report updated evaluation
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the neural network to a file

        parameters:
            filename: path to the file to save the network to

        Returns:
            None
        """
        import pickle

        # Validate file name
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        # Save the network to a file
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def load(filename):
        """
        Loads a deep neural network from a file

        parameters:
            filename: path to the file to load the network from

        Returns:
            DeepNeuralNetwork: the loaded network
        """
        import pickle

        # Validate file name
        if not filename.endswith(".pkl"):
            return None

        # Load the network from a file
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
