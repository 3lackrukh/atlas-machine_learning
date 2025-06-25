#!/usr/bin/env python3
"""Module for slicing DataFrame columns and rows"""
import pandas as pd


def slice(df):
    """
    Takes a pd.DataFrame as input and performs the following:
        - Extracts columns High, Low, Close, and Volume_(BTC)
        - Selects every 60th row

    Parameters:
        df: pd.DataFrame to slice

    Returns:
        The sliced pd.DataFrame
    """
    # Extract the columns High, Low, Close, and Volume_(BTC)
    selected_columns = ['High', 'Low', 'Close', 'Volume_(BTC)']
    df_sliced = df[selected_columns]

    # Select every 60th row
    result = df_sliced.iloc[::60]

    return result
