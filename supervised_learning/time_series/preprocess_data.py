#!/usr/bin/env python3
"""
Module for preprocessing Bitcoin time series data from Coinbase and Bitstamp
for forecasting applications.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller


def load_and_merge_data(coinbase_path, bitstamp_path):
    """
    Loads and merges Bitcoin data from Coinbase and Bitstamp exchanges.
    
    Args:
        coinbase_path: Path to Coinbase CSV file
        bitstamp_path: Path to Bitstamp CSV file
        
    Returns:
        Merged DataFrame with data from both exchanges
    """
    # Load data from both exchanges
    coinbase_df = pd.read_csv(coinbase_path)
    bitstamp_df = pd.read_csv(bitstamp_path)
    
    print(f"Coinbase data shape: {coinbase_df.shape}")
    print(f"Bitstamp data shape: {bitstamp_df.shape}")
    
    # Add source column to track the exchange
    coinbase_df['exchange'] = 'coinbase'
    bitstamp_df['exchange'] = 'bitstamp'
    
    # Combine data from both exchanges
    merged_df = pd.concat([coinbase_df, bitstamp_df], ignore_index=True)
    
    # Convert Unix timestamp to datetime
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'], unit='s')
    
    # Sort by timestamp
    merged_df.sort_values('Timestamp', inplace=True)
    
    return merged_df


def check_stationarity(series, name="Series"):
    """
    Performs Augmented Dickey-Fuller test to check stationarity.
    
    Args:
        series: Time series to test
        name: Name of the series for reporting
        
    Returns:
        Boolean indicating if series is stationary
    """
    result = adfuller(series.dropna())
    
    print(f"ADF Test for {name}")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
    
    is_stationary = result[1] <= 0.05
    print(f"Series is {'stationary' if is_stationary else 'non-stationary'}")
    
    return is_stationary


def clean_data(df):
    """
    Cleans the data by handling missing values and outliers.
    
    Args:
        df: DataFrame containing Bitcoin data
        
    Returns:
        Cleaned DataFrame
    """
    # Check for missing values
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    
    # Drop rows with missing values
    df_clean = df.dropna().copy()
    
    # Set timestamp as index for time series analysis
    df_clean.set_index('Timestamp', inplace=True)
    
    # Check for and replace infinity values with NaN
    #numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
    
    # Replace inf/-inf with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN again after replacing inf values
    df_clean.dropna(inplace=True)
    
    # Remove outliers using IQR method
    price_cols = ['Open', 'High', 'Low', 'Close', 'Weighted_Price']
    
    for col in price_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Filter out extreme outliers
        df_clean = df_clean[~((df_clean[col] < (Q1 - 3 * IQR)) | 
                              (df_clean[col] > (Q3 + 3 * IQR)))]
    
    print(f"Shape after cleaning: {df_clean.shape}")
    
    return df_clean


def resample_hourly(df):
    """
    Resamples data to hourly frequency for the forecasting task.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Hourly resampled DataFrame
    """
    # Define aggregation functions for different columns
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    }
    
    # Resample to hourly frequency
    df_hourly = df.resample('1h').agg(agg_dict)
    
    # Drop any remaining NaN values
    df_hourly.dropna(inplace=True)
    
    # Stationarity checks
    for col in [
        'Open', 'High', 'Low', 'Close',
        'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']:
        check_stationarity(df_hourly[col], col)
    
    print(f"Hourly resampled data shape: {df_hourly.shape}")
    
    return df_hourly

def make_stationary_features(df_hourly):
    """Transform non-stationary series into stationary ones"""
    df_stationary = df_hourly.copy()
    
    # Transform price series - these all need similar treatments
    price_cols = ['Open', 'High', 'Low', 'Close', 'Weighted_Price']
    
    for col in price_cols:
        # Log returns (recommended for price data)
        df_stationary[f'{col}_log_ret'] = np.log(df_stationary[col]).diff()
    
    # Drop NaN values created by differencing
    df_stationary.dropna(inplace=True)
    
    # Verify transformations worked
    print("\n--- CHECKING STATIONARITY OF TRANSFORMED SERIES ---")
    for col in price_cols:
        check_stationarity(df_stationary[f'{col}_log_ret'], f"{col} (Log Returns)")
    
    return df_stationary


def engineer_features(df):
    """
    Creates new features to improve time series forecasting.
    
    Args:
        df: Cleaned DataFrame with Bitcoin data
        
    Returns:
        DataFrame with engineered features
    """
    # Create a copy instead of modifying the original
    df_feat = df.copy()
    
    # Extract time-based features
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek
    
    # Calculate price changes
    df_feat['price_change'] = df_feat['Close'].pct_change()
    df_feat['stationary_price_change'] = df_feat['Close_log_ret'].pct_change()
    
    # Calculate rolling statistics (moving averages)
    #df_feat['ma_10min'] = df_feat['Close'].rolling('10min').mean()
    #df_feat['ma_30min'] = df_feat['Close'].rolling('30min').mean()
    df_feat['ma_1hour'] = df_feat['Close'].rolling('1h').mean()
    df_feat['stationary_ma_1hour'] = df_feat['Close_log_ret'].rolling('1h').mean()
    
    # Calculate volatility
    df_feat['volatility_1hour'] = df_feat['Close'].rolling('1h').std()
    df_feat['stationary_volatility_1hour'] = df_feat['Close_log_ret'].rolling('1h').std()
    
    # Calculate volume features
    df_feat['volume_change_(BTC)'] = df_feat['Volume_(BTC)'].pct_change()
    df_feat['volume_change_(Currency)'] = df_feat['Volume_(Currency)'].pct_change()
    
    # Replace inf/-inf with NaN
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values created by the calculations
    df_feat.dropna(inplace=True)
    
    # Verify no infinity values remain (can handle extreme but valid values)
    for col in df_feat.select_dtypes(include=[np.number]).columns:
        df_feat[col] = df_feat[col].clip(lower=-1e15, upper=1e15)
    
    return df_feat


def scale_data(df):
    """
    Scales the numerical features using MinMaxScaler.
    
    Args:
        df: DataFrame with hourly data
        
    Returns:
        DataFrame with scaled features and the scaler object
    """
    # Select numerical columns to scale
    numerical_cols = [col for col in df.columns 
                     if col not in ['hour', 'day_of_week', 'exchange']]
    
    # Make a copy of the data instead of modifying the original
    df_copy = df.copy()
    
    # Check for any remaining infinity or extreme values
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_copy.dropna(inplace=True)
    
    # Clip values to prevent extreme outliers from affecting scaling
    for col in numerical_cols:
        df_copy[col] = df_copy[col].clip(
            lower=df_copy[col].quantile(0.001),
            upper=df_copy[col].quantile(0.999)
        )
    
    # Verify data is clean before scaling
    print(f"Checking for infinite values before scaling: {np.any(np.isinf(df_copy[numerical_cols].values))}")
    print(f"Checking for NaN values before scaling: {np.any(np.isnan(df_copy[numerical_cols].values))}")
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Fit and transform the selected columns
    scaled_data = scaler.fit_transform(df_copy[numerical_cols])
    
    # Create a new DataFrame with scaled data
    df_scaled = pd.DataFrame(
        scaled_data, 
        columns=numerical_cols,
        index=df_copy.index
    )
    
    # Stationarity check after scaling
    for col in [
        'Open_log_ret', 'High_log_ret', 'Low_log_ret', 'Close_log_ret',
        'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']:
        check_stationarity(df_scaled[col], col)
    
    # Add non-scaled columns back
    for col in df_copy.columns:
        if col not in numerical_cols:
            df_scaled[col] = df_copy[col]
    
    return df_scaled, scaler, numerical_cols


def create_sequences(df, seq_length=24, target_col='Close'):
    """
    Creates sequences for time series forecasting.
    
    Args:
        df: DataFrame with scaled features
        seq_length: Input sequence length in hours (window size)
        target_col: Target column to predict
        
    Returns:
        Input sequences (X) and target values (y)
    """
    X, y = [], []
    
    # Get all feature columns
    feature_cols = df.columns.tolist()
    
    # Get index of the target column
    target_idx = feature_cols.index(target_col)
    
    # Convert DataFrame to numpy array
    data = df.values
    
    # Create sequences
    for i in range(len(data) - seq_length - 1):
        # Input sequence (past 24 hours)
        X.append(data[i:i+seq_length])
        
        # Target (close price for the next hour)
        y.append(data[i+seq_length][target_idx])
    
    return np.array(X), np.array(y)


def save_preprocessed_data(X, y, scaler, feature_cols, output_dir='preprocessed_data'):
    """
    Saves preprocessed data for later use.
    
    Args:
        X: Input sequences
        y: Target values
        scaler: Fitted scaler object
        feature_cols: List of feature column names
        output_dir: Output directory
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input sequences and targets
    np.save(os.path.join(output_dir, 'X_btc.npy'), X)
    np.save(os.path.join(output_dir, 'y_btc.npy'), y)
    
    # Save scaler and feature columns
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(output_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"Saved preprocessed data to {output_dir}")
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")


def main():
    """
    Main function to preprocess Bitcoin data.
    """
    # Paths to the data
    coinbase_path = 'coinbase.csv'
    bitstamp_path = 'bitstamp.csv'
    
    print("Loading and merging data...")
    df = load_and_merge_data(coinbase_path, bitstamp_path)
    
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    print("Resampling to hourly frequency...")
    df_hourly = resample_hourly(df_clean)
    
    print("Transforming non-stationary series...")
    df_stationary = make_stationary_features(df_hourly)
    
    print("Engineering features...")
    df_feat = engineer_features(df_stationary)
    
    print("Scaling data...")
    df_scaled, scaler, feature_cols = scale_data(df_feat)
    
    print("Creating sequences for time series forecasting...")
    X, y = create_sequences(df_scaled, seq_length=24, target_col='Close')
    
    print("Saving preprocessed data...")
    save_preprocessed_data(X, y, scaler, feature_cols)
    
    print("Preprocessing completed!")


if __name__ == '__main__':
    main()