#!/usr/bin/env python3
"""Module for loading data from file as pandas DataFrame"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pd.DataFrame.

    Parameters:
        filename: (str) path for file to load from
        delimiter: (str) The column separator

    Returns:
        The loaded pd.DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
