#!/usr/bin/env python3
""" Module defines the train_model method """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a given neural network model using
    mini-batch gradient descent and early stopping

    Parameters:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
            m: number of data points
            nx: number of input features
        labels: one-hot numpy.ndarray labels for the input data
        batch_size: integer batch size for gradient descent
        epochs: integer of passes through data for gradient descent
        validation_data: data to validate the model with, if not None
        early_stopping: boolean determines whether to stop training early
        patience: integer of epochs to wait before stopping training
        verbose: boolean determines whether to print training progress
        shuffle: boolean determines whether to shuffle the data

    Returns:
        the History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience))

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle, callbacks=callbacks)
