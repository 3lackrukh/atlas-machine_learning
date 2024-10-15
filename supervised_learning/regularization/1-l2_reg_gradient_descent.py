#!/usr/bin/env python3
""" Module defines the l2_reg_gradient_descent method """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Uses tanh activation for hidden layers and softmax for output layer

    Parameters:
        Y: One-hot numpy.ndarray of shape (classes, m)
            containing the one-hot labels for the data.
            classes: number of classes
            m: number of data points
        weights: Dictionary of weights and biases of the neural network.
        cache: Dictionary of outputs of each layer of the neural network.
        alpha: Floating point learning rate.
        lambtha: floating point L2 regularization parameter.
        L: Integer of layers in the neural network.
    """
    # Retrieve number of examples
    m = Y.shape[1]

    # Softmax for output layer
    dZ = cache[f'A{L}'] - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer - 1}']

        # Calculate gradients
        dW = np.matmul(dZ, A_prev.T) / m
        # Add regularization term
        dW += (lambtha / m) * weights[f'W{layer}']

        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.matmul(weights[f'W{layer}'].T, dZ)

        # Update dz for the next iteration
        if layer > 1:
            dZ = dA_prev * (1 - A_prev ** 2)

        # Update weights and biases
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db
