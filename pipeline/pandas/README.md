# Pandas Project

This project implements various pandas operations and data manipulation techniques using the coinbase and bitstamp cryptocurrency datasets.

## Description

Implementation of key pandas functionality including DataFrame creation, data loading, manipulation, analysis, and visualization using cryptocurrency time series data.

## Requirements

- Python 3.9
- Ubuntu 20.04 LTS
- numpy (version 1.25.2)
- pandas (version 2.2.2)
- matplotlib (for visualization)
- pycodestyle (version 2.11.1)

## Installation

Install pandas:
```bash
pip install --user pandas==2.2.2
```

## Files

### Core Functionality
- `0-from_numpy.py` - Creates DataFrame from numpy array with alphabetical columns
- `1-from_dictionary.py` - Creates DataFrame from dictionary with custom index
- `2-from_file.py` - Loads data from file with specified delimiter
- `3-rename.py` - Renames Timestamp column and converts to datetime
- `4-array.py` - Converts DataFrame columns to numpy array
- `5-slice.py` - Extracts specific columns and selects every 60th row
- `6-flip_switch.py` - Sorts data and transposes DataFrame
- `7-high.py` - Sorts DataFrame by High price in descending order
- `8-prune.py` - Removes rows with NaN values in Close column
- `9-fill.py` - Fills missing values in various columns
- `10-index.py` - Sets Timestamp column as DataFrame index
- `11-concat.py` - Concatenates DataFrames with specific conditions
- `12-hierarchy.py` - Handles hierarchical indexing and MultiIndex operations
- `13-analyze.py` - Computes descriptive statistics
- `14-visualize.py` - Visualizes data with transformations and plotting

## Learning Objectives

### General Concepts
- Understanding pandas DataFrame and Series data structures
- Creating DataFrames from various sources (numpy arrays, dictionaries, files)
- Performing indexing and selection operations
- Using hierarchical indexing with MultiIndex
- Slicing and filtering DataFrames
- Reassigning columns and sorting data
- Using boolean logic for filtering
- Merging, concatenating, and joining DataFrames
- Computing statistical information
- Visualizing DataFrames

### Specific Skills
- **Data Loading**: Reading CSV files with different delimiters
- **Data Cleaning**: Handling missing values, removing NaN entries
- **Data Transformation**: Converting timestamps, renaming columns
- **Data Analysis**: Computing statistics, grouping data
- **Data Visualization**: Creating plots with matplotlib
- **Time Series**: Working with datetime data and resampling
- **MultiIndex**: Managing hierarchical data structures

## Dataset

The project uses cryptocurrency datasets:
- `coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv` - Coinbase Bitcoin data
- `bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv` - Bitstamp Bitcoin data

These datasets contain timestamp, price (Open, High, Low, Close), volume, and weighted price information.

## Usage Examples

### Creating DataFrame from Numpy Array
```python
import numpy as np
from_numpy = __import__('0-from_numpy').from_numpy

A = np.random.randn(5, 8)
df = from_numpy(A)
```

### Loading Data from File
```python
from_file = __import__('2-from_file').from_file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
```

### Analyzing Data
```python
analyze = __import__('13-analyze').analyze
stats = analyze(df)
```

### Visualizing Data
```python
# Run the visualization script
./14-visualize.py
```

## Key Pandas Concepts Covered

1. **DataFrame Creation**: From arrays, dictionaries, and files
2. **Indexing**: Label-based and position-based selection
3. **Data Cleaning**: Handling missing values and data types
4. **Data Manipulation**: Sorting, filtering, and transforming
5. **Time Series**: Working with datetime data
6. **Aggregation**: Grouping and statistical operations
7. **Visualization**: Plotting time series data
8. **MultiIndex**: Hierarchical indexing operations

## Best Practices

- Always create copies when modifying DataFrames to avoid side effects
- Use appropriate data types for efficiency
- Handle missing values explicitly
- Document your data transformations
- Use vectorized operations when possible
- Consider memory usage when working with large datasets

## Testing

Each module can be tested using the provided main files:
- `0-main.py` through `14-main.py` (where applicable)

Run tests to verify functionality:
```bash
python3 0-main.py
python3 1-main.py
# ... etc
``` 