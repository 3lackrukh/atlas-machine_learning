#!/usr/bin/env python3
"""Module for setting Timestamp column as index"""


def index(df):
    """
    Takes a pd.DataFrame and performs the following:
        - Sets the Timestamp column as the index of the dataframe

    Parameters:
        df: pd.DataFrame containing a Timestamp column

    Returns:
        The modified pd.DataFrame
    """
    # Set Timestamp column as index
    result = df.set_index('Timestamp')

    return result
