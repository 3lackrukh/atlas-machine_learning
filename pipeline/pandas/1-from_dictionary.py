#!/usr/bin/env python3
"""Module for creating pandas DataFrame from dictionary"""
import pandas as pd

# Create dictionary with the specified data
data = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}

# Create DataFrame with custom row labels
df = pd.DataFrame(data, index=['A', 'B', 'C', 'D'])
