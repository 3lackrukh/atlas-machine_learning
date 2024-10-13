#!/usr/bin/env python3
""" Module defines the create_confusion_matrix method """
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Parameters:
        labels: one-hot numpy.ndarray of shape (m, classes)
            containing correct labels for each data point
            m: integer count of data points
            classes: integer count of classes
        logits: one-hot numpy.ndarray of shape (classes, classes)
            containing the prediceted labels

    Returns:
        numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and
            column indices represent the predicted labels.
    """
    return np.matmul(labels.T, logits)
