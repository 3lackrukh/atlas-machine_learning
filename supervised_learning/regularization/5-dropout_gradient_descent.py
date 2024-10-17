#!/usr/bin/env python3
""" Module defines the dropout_gradient_descent method """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with
    Dropout regularization using gradient descent.

    Hidden layers utilize tanh activation
    Output layer utilizes softmax activation

    Parameters:
        Y: One-Hot ndarray of shape (classes, m)
        containing labels for the data.
            classes: integer of classes
            m: integer of data points
        weights: Dictionary of weights and biases for the network
        cache: Dictionary of values for each layer in the network.
        alpha: floating point learning rate for gradient descent.
        keep_prob: floating point probability that a node will be kept.
        L: integer of layers in the network.

    Returns:
        Nothing (Updates weights in place)
    """
    m = Y.shape[1]

    # For the output layer (softmax activation)
    dz = cache[f'A{L}'] - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f'A{layer - 1}']

        # Apply dropout before gradient descent
        if layer > 1:
            # For hidden layers (tanh activation with dropout)
            dA = np.matmul(weights[f'W{layer}'].T, dz)

            # Apply dropout only if the mask exists for this layer
            if f'D{layer-1}' in cache:
                dA *= cache[f'D{layer-1}']  # Apply dropout mask
                dA /= keep_prob  # Scale the values

        # Calculate gradients
        dW = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        # Update weights and biases
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * db

        if layer > 1:
            # Derivative of tanh activation
            dz = dA * (1 - (A_prev ** 2))
