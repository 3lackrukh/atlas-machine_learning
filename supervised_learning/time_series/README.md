# Bitcoin Price Forecasting

This project implements a time series forecasting model to predict Bitcoin (BTC) prices using Recurrent Neural Networks (RNNs).

## Overview

The goal is to use historical Bitcoin price data from Coinbase and Bitstamp exchanges to predict the closing price of Bitcoin at the end of the next hour, based on the previous 24 hours of data.

## Dataset
The dataset must be extracted from the following google drive links
and saved to the root of the project repository as coinbasey.csv and bitstamp.csv respectively:

https://drive.google.com/file/d/16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE/view

https://drive.google.com/file/d/15A-rLSrfZ0td7muSrYHy0WX9ZqrMweES/view

The datasets contain Bitcoin trading data from two major cryptocurrency exchanges:
- Coinbase 
- Bitstamp

Each row in the datasets represents a 60-second time window with the following information:
- Start time (Unix timestamp)
- Open price (USD)
- High price (USD)
- Low price (USD)
- Close price (USD)
- BTC volume
- USD volume
- Volume-weighted average price (VWAP)

## Project Structure

- `preprocess_data.py`: Script to preprocess the raw Bitcoin data
- `forecast_btc.py`: Script to build, train and validate the RNN model
- `preprocessed_data/`: Directory containing preprocessed data
- `btc_forecast_model.h5`: Trained model file
- `btc_predictions.png`: Visualization of model predictions
- `training_history.png`: Visualization of training process

## Technical Approach

### Data Preprocessing

The preprocessing pipeline includes:
1. Loading and merging data from both exchanges
2. Cleaning (handling missing values and outliers)
3. Feature engineering (time-based features, rolling statistics, etc.)
4. Resampling to hourly frequency
5. Scaling numerical features
6. Creating sequences for time series forecasting (24-hour sliding windows)

### Model Architecture

The model uses a stacked RNN architecture with:
- Two LSTM/GRU/SimpleRNN layers
- Dropout layers for regularization
- Dense layers for the final prediction
- Mean Squared Error (MSE) as the loss function

### Training and Evaluation

The model is trained using:
- A training set (~70% of data)
- A validation set (~10% of data)
- Early stopping to prevent overfitting
- Learning rate reduction on plateau

Evaluation metrics include:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Usage

1. Preprocess the raw data:
```
python preprocess_data.py
```

2. Train and evaluate the model:
```
python forecast_btc.py
```

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Pandas 2.2.2
- Matplotlib
- Scikit-learn

## Learning Outcomes

This project demonstrates understanding of:
- Time series forecasting concepts
- Stationarity in time series data
- Data preprocessing for time series
- TensorFlow data pipelines
- RNN architectures (LSTM, GRU, SimpleRNN)
- Model evaluation for time series forecasting