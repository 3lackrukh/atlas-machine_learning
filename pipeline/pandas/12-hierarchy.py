#!/usr/bin/env python3
"""Module for hierarchical indexing and concatenation"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Takes two pd.DataFrames and performs the following:
        - Rearranges the MultiIndex so that Timestamp is the first level
        - Concatenates the bitstamp and coinbase tables
            from timestamps 1417411980 to 1417417980, inclusive
        - Adds keys to the data,
            Labeling rows from df2 as bitstamp
            and rows from df1 as coinbase
        - Ensures data is displayed in chronological order

    Parameters:
        df1: First pd.DataFrame (coinbase)
        df2: Second pd.DataFrame (bitstamp)

    Returns:
        The concatenated pd.DataFrame with rearranged MultiIndex
    """
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter timestamps 1417411980 to 1417417980 INCLUSIVE
    df1_filtered = df1_indexed[(df1_indexed.index >= 1417411980) &
                               (df1_indexed.index <= 1417417980)]
    df2_filtered = df2_indexed[(df2_indexed.index >= 1417411980) &
                               (df2_indexed.index <= 1417417980)]

    # Concatenate with keys bitstamp and coinbase
    concatenated = pd.concat([df2_filtered, df1_filtered],
                             keys=['bitstamp', 'coinbase'])

    # Rearrange MultiIndex so Timestamp is first level
    result = concatenated.swaplevel(0, 1).sort_index(level=0)

    return result
