#!/usr/bin/env python3
"""Module for sorting DataFrame by High price"""


def high(df):
    """
    Takes a pd.DataFrame as input and performs the following:
        - Sorts the DataFrame by High price in descending order

    Parameters:
        df: pd.DataFrame to sort

    Returns:
        The sorted pd.DataFrame
    """
    # Sort by High price in descending order
    result = df.sort_values(by='High', ascending=False)

    return result
