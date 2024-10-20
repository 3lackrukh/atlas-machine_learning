#!/usr/bin/env python3
""" Module defines the train_model method """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a given neural network model using mini-batch gradient descent

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
            m: number of data points
            nx: number of input features
        labels: one-hot numpy.ndarray labels for the input data
        batch_size: integer batch size for gradient descent
        epochs: integer of passes through data for gradient descent
        verbose: boolean determines whether to print training progress
        shuffle: boolean determines whether to shuffle the data

    Returns:
        the History object generated after training the model
    """
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle)
    