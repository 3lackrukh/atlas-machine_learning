#!/usr/bin/env python3
""" Module defines the dropout_forward_prop method """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Layers implement tanh activation
    Output layer implements softmax activation

    Parameters:
    X: numpy.ndarray of shape (nx, m) containing input data
        nx: integer of input features
        m: integer of data points
    weights: dictionary of weights and biases
    L: integer of layers in the network
    keep_prob: floating point probability of keeping a neuron active

    Returns:
    dict: Dictionary containing the output of each layer and
        the dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X
    A = X

    for layer in range(1, L + 1):
        W = weights[f'W{layer}']
        b = weights[f'b{layer}']

        # Calculate current layer output
        z = np.matmul(W, A) + b

        if layer == L:
            # Softmax activation for output layer
            t = np.exp(z)
            cache[f'A{layer}'] = t / np.sum(t, axis=0, keepdims=True)
        else:
            # tanh activation for hidden layers
            A = np.tanh(z)

            # Apply dropout
            drop = np.random.rand(* A.shape) < keep_prob
            A *= drop
            A /= keep_prob

            cache[f'A{layer}'] = A
            cache[f'D{layer}'] = drop.astype(int)

    return cache
