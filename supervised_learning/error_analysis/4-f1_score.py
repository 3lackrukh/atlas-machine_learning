#!/usr/bin/env python3
""" Module defines the f1_score method """
import numpy as np


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix

    Parameters:
        Confusion: numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels
            classes: the number of classes

    Returns: numpy.ndarray of shape (classes,)
            containing the sensitivity of each class
    """
    sensitivity = __import__('1-sensitivity').sensitivity(confusion)
    precision = __import__('2-precision').precision(confusion)

    # Calculate and return f1_score
    return 2 * (sensitivity * precision) / (sensitivity + precision)
