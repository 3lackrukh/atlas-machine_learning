#!/usr/bin/env python3
""" Module defines the sensitivity method """
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Parameters:
        confusion: numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels
            classes: the number of classes

    Returns:
        numpy.ndarray of shape (classes,)
            containing the sensitivity of each class
    """
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    return true_positives / (true_positives + false_negatives)
