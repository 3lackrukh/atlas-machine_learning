#!/usr/bin/env python3
""" Module defines create_mini_batches method """
shuffle_data = __import__('2-shuffle_data').shuffle_data

def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network.

    Parameters:
        X: numpy.ndarray - input data.
        Y: numpy.ndarray - labels for input data.
        batch_size: int - size of mini-batches.

    Returns:
        batches: list of tuples (X_batch, Y_batch)
            X_batch: Input data
            Y_batch: Labels for input data
    """
    m = X.shape[0]
    batches = []
    X, Y = shuffle_data(X, Y)

    for i in range(0, m, batch_size):
        # Ensure the last batch includes all remaining data
        if i + batch_size > m:
            batch_size = m - i
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        batches.append((X_batch, Y_batch))

    return batches
