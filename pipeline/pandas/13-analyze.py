#!/usr/bin/env python3
"""Module for computing descriptive statistics"""


def analyze(df):
    """
    Takes a pd.DataFrame and performs the following:
        - Computes descriptive statistics for all columns
            except the Timestamp column.

    Parameters:
        df: pd.DataFrame to analyze

    Returns:
        A new pd.DataFrame containing the statistics
    """
    # Exclude the Timestamp column if it exists
    if 'Timestamp' in df.columns:
        columns_to_analyze = df.drop(columns=['Timestamp'])
    else:
        columns_to_analyze = df

    # Compute descriptive statistics
    result = columns_to_analyze.describe()

    return result
