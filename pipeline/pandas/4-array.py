#!/usr/bin/env python3
"""Module for converting DataFrame columns to numpy array"""
import pandas as pd


def array(df):
    """
    Takes a pd.DataFrame as input and performs the following:
        - Selects the last 10 rows of High and Close columns
        - Converts selected values to a numpy.ndarray

    Parameters:
        df: pd.DataFrame containing columns named High and Close

    Returns:
        numpy.ndarray with the selected values
    """
    # Select the last 10 rows of High and Close columns
    selected_data = df[['High', 'Close']].tail(10)

    # Convert to numpy array
    result = selected_data.to_numpy()

    return result
