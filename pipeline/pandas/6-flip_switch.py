#!/usr/bin/env python3
"""Module for sorting and transposing DataFrame"""


def flip_switch(df):
    """
    Takes a pd.Dataframe and performs the following:
        - Sorts the data in reverse chronological order
        - Transposes the sorted dataframe

    Parameters:
        df: pd.DataFrame to transform

    Returns:
        The transformed pd.DataFrame
    """
    # Sort the data in reverse chronological order
    # Assuming the first column is Timestamp, sort by it in descending order
    df_sorted = df.sort_values(by=df.columns[0], ascending=False)

    # Transpose the sorted dataframe
    result = df_sorted.T

    return result
