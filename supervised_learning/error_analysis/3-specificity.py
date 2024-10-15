#!/usr/bin/env python3
""" Module defines the specificity method """
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Parameters:
        confusion: numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels
            classes: the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
        of each class
    """
    total_samples = np.sum(confusion)
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # Calculate true_negatives for each class
    true_negatives = total_samples \
        - (false_positives + false_negatives + true_positives)

    # Calculate and return specificity of classes
    return true_negatives / (true_negatives + false_positives)
