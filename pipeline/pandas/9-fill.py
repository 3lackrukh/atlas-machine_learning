#!/usr/bin/env python3
"""Module for filling missing values in DataFrame"""

import pandas as pd


def fill(df):
    """
    Removes Weighted_Price column and fills missing values in various columns.

    Args:
        df: pd.DataFrame to modify

    Returns:
        The modified pd.DataFrame
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Remove the Weighted_Price column
    df_copy = df_copy.drop(columns=['Weighted_Price'])
    
    # Fill missing values in Close column with the previous row's value
    df_copy['Close'] = df_copy['Close'].fillna(method='ffill')
    
    # Fill missing values in High, Low, and Open columns with the corresponding Close value
    df_copy['High'] = df_copy['High'].fillna(df_copy['Close'])
    df_copy['Low'] = df_copy['Low'].fillna(df_copy['Close'])
    df_copy['Open'] = df_copy['Open'].fillna(df_copy['Close'])
    
    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    df_copy['Volume_(BTC)'] = df_copy['Volume_(BTC)'].fillna(0)
    df_copy['Volume_(Currency)'] = df_copy['Volume_(Currency)'].fillna(0)
    
    return df_copy 