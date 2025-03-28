#!/usr/bin/env python3
"""
Module for forecasting Bitcoin prices using RNN-based models.
"""

import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, SimpleRNN, Dense, Dropout
import matplotlib.pyplot as plt


def load_preprocessed_data(data_dir='preprocessed_data'):
    """
    Loads preprocessed data for Bitcoin price forecasting.
    
    Args:
        data_dir: Directory containing preprocessed data
        
    Returns:
        Input sequences, target values, scaler, and feature column names
    """
    # Load the sequences and targets
    X = np.load(os.path.join(data_dir, 'X_btc.npy'))
    y = np.load(os.path.join(data_dir, 'y_btc.npy'))
    
    # Load the scaler and feature columns
    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(data_dir, 'feature_cols.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, scaler, feature_cols


def split_data(X, y, test_size=0.2, val_size=0.1):
    """
    Splits data into training, validation, and test sets.
    
    Args:
        X: Input sequences
        y: Target values
        test_size: Proportion of data for testing
        val_size: Proportion of non-test data for validation
        
    Returns:
        Training, validation, and test data
    """
    # Calculate split indices
    n_samples = len(X)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split data chronologically
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_tf_dataset(X, y, batch_size=32, shuffle_buffer=None):
    """
    Creates a TensorFlow Dataset for efficient training.
    
    Args:
        X: Input sequences
        y: Target values
        batch_size: Batch size
        shuffle_buffer: Buffer size for shuffling
        
    Returns:
        TensorFlow Dataset
    """
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle if specified
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def build_rnn_model(input_shape, rnn_type='lstm'):
    """
    Builds an RNN model for time series forecasting.
    
    Args:
        input_shape: Shape of input sequences (seq_length, n_features)
        rnn_type: Type of RNN layer ('lstm', 'gru', or 'simple_rnn')
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First RNN layer
    if rnn_type == 'lstm':
        model.add(Bidirectional(
            LSTM(32, activation='relu', return_sequences=False), input_shape=input_shape))
    elif rnn_type == 'gru':
        model.add(Bidirectional(
            GRU(32, activation='relu', return_sequences=False), input_shape=input_shape))
    elif rnn_type == 'simple_rnn':
        model.add(Bidirectional(
            SimpleRNN(32, activation='relu', return_sequences=False), input_shape=input_shape))
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")
    
    #add(Dropout(0.2))
    
    # Second RNN layer
    #if rnn_type == 'lstm':
    #    model.add(LSTM(64, activation='relu'))
    #elif rnn_type == 'gru':
    #    model.add(GRU(64, activation='relu'))
    #elif rnn_type == 'simple_rnn':
    #    model.add(SimpleRNN(64, activation='relu'))
    
    #model.add(Dropout(0.2))
    
    # Dense layers
    #model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for price prediction
    
    # Compile the model with MSE loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


def train_model(model, train_dataset, val_dataset, epochs=50):
    """
    Trains the model.
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Maximum number of epochs
        
    Returns:
        Training history
    """
    # Define callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        # Reduce learning rate when training plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history


def evaluate_model(model, X_test, y_test, scaler, feature_cols, target_col):
    """
    Evaluates the model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test input sequences
        y_test: Test target values
        scaler: Scaler used to normalize the data
        feature_cols: Feature column names
        target_col: Target column name for evaluation
        
    Returns:
        Evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics on scaled data
    mse = np.mean((y_pred.flatten() - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred.flatten() - y_test))
    
    print(f"Scaled MSE (on log returns): {mse:.6f}")
    print(f"Scaled RMSE (on log returns): {rmse:.6f}")
    print(f"Scaled MAE (on log returns): {mae:.6f}")
    
    # Inverse transform predictions and actual values
    # We need to create dummy arrays with all features
    target_idx = feature_cols.index(target_col)
    close_idx = feature_cols.index('Close')
    dummy_array = np.zeros((len(y_test), len(feature_cols)))
    
    # Set the Close column values
    dummy_array[:, target_idx] = y_test
    y_test_dummy = dummy_array.copy()
    
    dummy_array[:, target_idx] = y_pred.flatten()
    y_pred_dummy = dummy_array.copy()
    
    # Inverse transform to get original log returns
    y_test_log_ret = scaler.inverse_transform(y_test_dummy)[:, target_idx]
    y_pred_log_ret = scaler.inverse_transform(y_pred_dummy)[:, target_idx]
    
    # Convert from log returns to actual prices
    last_price = None
    last_scaled_close = X_test[0, -1, close_idx]
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0, close_idx] = last_scaled_close
    last_price = scaler.inverse_transform(dummy)[0, close_idx]
    
    print(f"Starting price for conversion: ${last_price:.2f}")
    
    # Convert log returns to actual prices
    y_test_prices = []
    y_pred_prices = []
    
    current_price = last_price
    for i in range(len(y_test_log_ret)):
        # Convert actual log returns to prices
        next_true_price = current_price * np.exp(y_test_log_ret[i])
        y_test_prices.append(next_true_price)
        
        # Convert predicted log returns to prices
        next_pred_price = current_price * np.exp(y_pred_log_ret[i])
        y_pred_prices.append(next_pred_price)
        
        # Use true price for next prediction to prevent error accumulation
        current_price = next_true_price
    
    # Calculate metrics on original scale
    price_mse = np.mean((np.array(y_pred_prices) - np.array(y_test_prices)) ** 2)
    price_rmse = np.sqrt(price_mse)
    price_mae = np.mean(np.abs(np.array(y_pred_prices) - np.array(y_test_prices)))
    
    print("\nMetrics on actual prices:")
    print(f"MSE: {price_mse:.2f}")
    print(f"RMSE: {price_rmse:.2f}")
    print(f"MAE: {price_mae:.2f}")
    
    # Calculate percentage metrics
    mape = np.mean(np.abs((np.array(y_test_prices) - np.array(y_pred_prices)) / np.array(y_test_prices))) * 100
    print(f"MAPE: {mape:.2f}%")
    
    # Visualize predictions
    plot_predictions(y_test_prices, y_pred_prices)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'price_mse': price_mse,
        'price_rmse': price_rmse,
        'price_mae': price_mae,
        'mape': mape
    }


def plot_predictions(y_true, y_pred, n_samples=100):
    """
    Plots actual vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_samples: Number of samples to plot
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    
    # Get last n_samples for better visualization
    n_samples = min(n_samples, len(y_true))
    
    plt.plot(y_true[-n_samples:], label='Actual')
    plt.plot(y_pred[-n_samples:], label='Predicted')
    
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('btc_predictions.png')
    plt.close()


def plot_training_history(history):
    """
    Plots training history.
    
    Args:
        history: Training history
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def main():
    """
    Main function for Bitcoin price forecasting.
    """
    print("Bitcoin Price Forecasting with RNNs")
    print("=================================\n")
    
    # Load preprocessed data
    X, y, scaler, feature_cols = load_preprocessed_data()
    
    # Define target column
    target_col = 'Close_log_ret'
    print(f"Using target column: {target_col} for evaluation")
    
    # Split data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=32, shuffle_buffer=1000)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=32)
    
    # Build model
    input_shape = (X.shape[1], X.shape[2])  # (seq_length, n_features)
    model = build_rnn_model(input_shape, rnn_type='lstm')
    
    # Print model summary
    model.summary()
    
    # Train model
    history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=50
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, scaler, feature_cols, target_col=target_col)
    
    # Save model
    model.save('btc_forecast_model.h5')
    
    print("Bitcoin price forecasting completed!")


if __name__ == '__main__':
    main()