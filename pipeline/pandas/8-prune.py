#!/usr/bin/env python3
"""Module for removing NaN values from Close column"""


def prune(df):
    """
    Takes a pd.DataFrame and performs the following:
        - Removes any entries where Close has NaN values

    Parameters:
        df: pd.DataFrame to clean

    Returns:
        The modified pd.DataFrame
    """
    # Remove rows where Close has NaN values
    result = df.dropna(subset=['Close'])

    return result
