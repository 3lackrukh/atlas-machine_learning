#!/usr/bin/env python3
"""Module for concatenating DataFrames with specific conditions"""
import pandas as pd
index = __import__('10-index')


def concat(df1, df2):
    """
    Takes two pd.DataFrames and performs the following:
        - Indexes both dataframes on their Timestamp columns
        - Includes all timestamps from df2
            up to and including timestamp 1417411920
        - Concatenates the selected rows from df2 to the top of df1
        - Adds keys to label the rows from df2 as bitstamp
            and the rows from df1 as coinbase

    Parameters:
        df1: First pd.DataFrame (coinbase)
        df2: Second pd.DataFrame (bitstamp)

    Returns:
        The concatenated pd.DataFrame
    """
    # Index dataframes on Timestamp
    df1_indexed = df1.set_index('Timestamp')
    df2_indexed = df2.set_index('Timestamp')

    # Include timestamps from df2 up to 1417411920 INCLUSIVE
    df2_filtered = df2_indexed[df2_indexed.index <= 1417411920]

    # Concatenate rows from df2 to the top of df1
    # Add keys to df2 as bitstamp and df1 as coinbase
    result = pd.concat([df2_filtered, df1_indexed],
                       keys=['bitstamp', 'coinbase'])

    return result
