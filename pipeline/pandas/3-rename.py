#!/usr/bin/env python3
"""Module for renaming and converting timestamp column"""

import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime and converts timestamp values to datetime.

    Args:
        df: pd.DataFrame containing a column named Timestamp

    Returns:
        The modified pd.DataFrame with only Datetime and Close columns
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Rename Timestamp column to Datetime
    df_copy = df_copy.rename(columns={'Timestamp': 'Datetime'})
    
    # Convert timestamp values to datetime values
    df_copy['Datetime'] = pd.to_datetime(df_copy['Datetime'], unit='s')
    
    # Display only the Datetime and Close column
    result = df_copy[['Datetime', 'Close']]
    
    return result 