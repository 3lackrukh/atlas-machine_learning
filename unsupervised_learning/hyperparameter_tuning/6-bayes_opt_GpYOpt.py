#!/usr/bin/env python3

import numpy as np
import GPyOpt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import datetime
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create a directory to store checkpoints
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')


# Load and prepare data
def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()


def create_model(learning_rate, first_layer_units, second_layer_units,
                 dropout_rate, l2_weight):
    model = Sequential()
    model.add(Dense(int(first_layer_units), activation='relu',
                    input_shape=(X_train.shape[1],),
                    kernel_regularizer=l2(l2_weight)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(second_layer_units), activation='relu',
                    kernel_regularizer=l2(l2_weight)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(hyperparams):
    # Unwrap hyperparameters
    learning_rate = float(hyperparams[:, 0])
    first_layer_units = int(hyperparams[:, 1])
    second_layer_units = int(hyperparams[:, 2])
    dropout_rate = float(hyperparams[:, 3])
    l2_weight = float(hyperparams[:, 4])
    batch_size = int(hyperparams[:, 5])

    # Create the model
    model = create_model(
        learning_rate=learning_rate,
        first_layer_units=first_layer_units,
        second_layer_units=second_layer_units,
        dropout_rate=dropout_rate,
        l2_weight=l2_weight
    )

    # Define callbacks
    checkpoint_filename = 'checkpoints/model_lr-{:.5f}_fu-{}_su-{}_dr-{:.3f}_l2-{:.6f}_bs-{}.h5'.format(
        learning_rate, first_layer_units, second_layer_units, dropout_rate, l2_weight, batch_size
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_filename,
                        monitor='val_accuracy', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate on test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Since GPyOpt minimizes, return negative accuracy
    return -accuracy


# Define the search space
bounds = [
    {'name': 'learning_rate', 'type': 'continuous',
     'domain': (0.0001, 0.01)},
    {'name': 'first_layer_units', 'type': 'discrete',
     'domain': (32, 64, 128, 256)},
    {'name': 'second_layer_units', 'type': 'discrete',
     'domain': (16, 32, 64, 128)},
    {'name': 'dropout_rate', 'type': 'continuous',
     'domain': (0.1, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous',
     'domain': (0.0001, 0.01)},
    {'name': 'batch_size', 'type': 'discrete',
     'domain': (16, 32, 64, 128)}
]
# Initialize progress bar
max_iter = 30
total_iters = max_iter + 5  # Account for initial design points
eval_count = 0
current_best = None
progbar = tqdm(total=total_iters, desc='Bayesian Optimization')


# Wrap the train_model function to update progress
def train_model_with_progress(hyperparams):
    global eval_count, current_best
    result = train_model(hyperparams)
    eval_count += 1

    # Update current best accuracy if necessary
    if current_best is None or -result > current_best:
        current_best = -result

    # Update progress bar with current status
    if eval_count <= total_iters:
        progbar.set_postfix({'Best Accuracy': f'{current_best:.4f}'
                             if current_best else 'N/A'})
        progbar.update(1)

    return result


# Define the Bayesian optimizer
optimizer = GPyOpt.methods.BayesianOptimization(
    f=train_model_with_progress,
    domain=bounds,
    model_type='GP',
    initial_design_numdata=5,
    initial_design_type='random',
    acquisition_type='EI',
    maximize=False,
    normalize_Y=True,
    verbosity=True
)

# Run the optimization
optimizer.run_optimization(max_iter=max_iter)
progbar.close()

# Get the results
best_hyperparams = optimizer.x_opt
best_value = -optimizer.fx_opt  # Convert back to accuracy

# Save the optimization report
with open('bayes_opt.txt', 'w') as f:
    f.write('Bayesian Optimization Report\n')
    f.write('==========================\n\n')
    f.write(f'Best hyperparameters found after {max_iter} iterations:\n')
    f.write(f'Learning rate: {best_hyperparams[0]:.6f}\n')
    f.write(f'First layer units: {int(best_hyperparams[1])}\n')
    f.write(f'Second layer units: {int(best_hyperparams[2])}\n')
    f.write(f'Dropout rate: {best_hyperparams[3]:.3f}\n')
    f.write(f'L2 weight: {best_hyperparams[4]:.6f}\n')
    f.write(f'Batch size: {int(best_hyperparams[5])}\n\n')
    f.write(f'Best validation accuracy: {best_value:.4f}\n\n')
    f.write('Optimization history:\n')
    for i, (params, value) in enumerate(zip(optimizer.X, -optimizer.Y)):
        f.write(f'Iteration {i+1}:\n')
        f.write(f'  Learning rate: {params[0]:.6f}\n')
        f.write(f'  First layer units: {int(params[1])}\n')
        f.write(f'  Second layer units: {int(params[2])}\n')
        f.write(f'  Dropout rate: {params[3]:.3f}\n')
        f.write(f'  L2 weight: {params[4]:.6f}\n')
        f.write(f'  Batch size: {int(params[5])}\n')
        f.write(f'  Validation accuracy: {value[0]:.4f}\n\n')

# Plot the convergence
optimizer.plot_convergence('convergence.png')
plt.tight_layout()
plt.savefig('convergence.png', dpi=300)
plt.show()

# Print final results
print('Optimization completed.')
print(f'Best hyperparameters: {best_hyperparams}')
print(f'Best test accuracy: {best_value:.4f}')
print('Saved optimization report to bayes_opt.txt')
print('Saved convergence plot to convergence.png')
