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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0  
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
