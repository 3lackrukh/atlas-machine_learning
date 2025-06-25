#!/usr/bin/env python3
"""Module for creating pandas DataFrame from numpy array"""
import pandas as pd
import numpy as np


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Parameters:
        array: np.ndarray from which to create the pd.DataFrame

    Returns:
        The newly created pd.DataFrrame
    """
    # Get the number of columns
    num_cols = array.shape[1]

    # Create column labels in alphabetical order (A, B, C, ...)
    # There will not be more than 26 columns
    columns = [chr(65 + i) for i in range(num_cols)]

    # Create DataFrame
    df = pd.DataFrame(array, columns=columns)

    return df
