#!/usr/bin/env python3
""" Module for training a CNN to classify CIFAR 10 dataset
    using a pre-built Keras Application for transfer learning """
from tensorflow import keras as K
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_training_history(history):
    """
    Plots training history including accuracy and loss curves.
    
    Parameters:
        history: History object returned by model.fit()
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots confusion matrix for the model predictions.
    
    Parameters:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (model output probabilities)
    """
    # Convert one-hot encoded labels back to class indices
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes,
                              target_names=class_names))

def plot_class_accuracy(y_true, y_pred):
    """
    Plots accuracy for each class.
    
    Parameters:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted labels (model output probabilities)
    """
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracies = []
    for i in range(10):
        mask = (y_true_classes == i)
        acc = np.mean(y_pred_classes[mask] == i)
        accuracies.append(acc)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, accuracies)
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.close()


def preprocess_data(X, Y):
    """
    Preprocesses the data for training.

    Parameters:
        X (numpy.ndarray): Input data of shape (m, 32, 32, 3)
                           containing the CIFAR 10 images.
        Y (numpy.ndarray): Labels for the input data.

    Returns:
        X_p (numpy.ndarray): Preprocessed input data of shape (m, 384, 384, 3).
        Y_p (numpy.ndarray): Preprocessed labels in one-hot encoded format.
    """
    # Scale pixel values
    X_p = K.applications.efficientnet_v2.preprocess_input(X)
    # Convert labels to one-hot encoded format
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    
    # Add callbacks for early stopping and model checkpoint
    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.001,
            patience=5,
            mode='max'
        ),
        K.callbacks.ModelCheckpoint(
            'cifar10.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    # Load and pre-process CIFAR-10 dataset
    print('Loading and preprocessing data...')
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)
    
    # Resize layer for input images
    # Attempt 1 using bicubic interpolation
    # Chosen for balanced detail preservation vs speed
    print('Initializing Resizer: bicubic interpolation')
    Resizer = K.layers.Resizing(384, 384, interpolation='bicubic')

    # Create base model for transfer learning
    # Attempt 1 EfficientNetV2S
    #   selected for parameter efficiency
    #   selected for progressive learning architecture
    # Load the EfficientNetV2S model with imagenet weights
    #   for size efficiency
    #   for more specialized tasks consider imagenet21k
    # no pooling for feature preservation
    print('Initializing base_model: EfficientNetV2S ...')
    base_model = K.applications.EfficientNetV2S(
        input_shape=(384, 384, 3),
        include_top=False,
        weights='imagenet',
        pooling='none',
        include_preprocessing=True
    )

    # Create model with resizer
    print('Creating key_model for feature extraction ...')
    key_model = K.Sequential([
        Resizer,
        base_model
    ])

    # Freeze the base model layers
    print('Freezing key_model layers')
    key_model.trainable = False
    
    def cbam_spatial_attention(x):
        """
        Implements CBAM spatial attention mechanism.

        Parameters:
            x: Input tensor
        
        Returns:
            Tensor with spatial attention applied
        """
        x_reshape = K.layers.Reshape((12, 12, 1280))(x)
        # Compute channel-wise average and max pool
        avg_pool = K.layers.GlobalAveragePooling2D(keepdims=True)(x_reshape)
        max_pool = K.layers.GlobalMaxPooling2D(keepdims=True)(x_reshape)

        mlp = K.Sequential([
            K.layers.Dense(1280 // 16, activation='relu'),
            K.layers.Dense(1280, activation='sigmoid')
        ])
        
        channel_attention = mlp(avg_pool) + mlp(max_pool)
        x_chan = x_reshape * channel_attention
        
        avg_spatial = K.layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x_chan)
        max_spatial = K.layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x_chan)
        spatial_concat = K.layers.Concatenate(axis=-1)([avg_spatial, max_spatial])
        
        spatial_attention = K.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
        x_spatial = x_chan * spatial_attention
        
        x_flat = K.layers.GlobalAveragePooling2D()(x_spatial)
        
        return x_flat
    

    # Check for saved features
    if os.path.exists('cifar10_features.h5'):
        print('Loading saved features ...')
        with h5py.File('cifar10_features.h5', 'r') as f:
            train_features = f['train_features'][:]
            test_features = f['test_features'][:]
            train_labels = f['train_labels'][:]
            test_labels = f['test_labels'][:]
        print('Features loaded!')
    else:
        # Compute and save features from frozen layers
        print('Computing features ...')
        train_features = key_model.predict(X_train_p)
        test_features = key_model.predict(X_test_p)
    
        # Save features for later use
        print('Saving features ...')
        with h5py.File('cifar10_features.h5', 'w') as f:
            f.create_dataset('train_features', data=train_features)
            f.create_dataset('test_features', data=test_features)
            f.create_dataset('train_labels', data=Y_train_p)
            f.create_dataset('test_labels', data=Y_test_p)
        print('Features saved!')
    

    # Create new top model
    print('Creating and training top_model ...')
    # Attempt 1 categorical softmax output layer to establish a baseline
    # Attempt 2 attention and classification layers
    feature_input=K.Input(shape=(12, 12, 1280,)) # Shape of EfficientNetV2S output
    #dense = K.layers.Dense(256, kernel_regularizer=K.regularizers.l2(0.01),
                           #kernel_initializer='he_normal')(feature_input)
    #bn = K.layers.BatchNormalization()(feature_input)
    #relu = K.layers.Activation('relu')(bn)
    #dropout = K.layers.Dropout(0.2)(bn)
    
    #dense = K.layers.Dense(256)(dropout)
    #bn = K.layers.BatchNormalization()(dense)
    #relu = K.layers.Activation('relu')(bn)
    #dropout = K.layers.Dropout(0.2)(relu)
    
    #dense = K.layers.Dense(128)(dropout)
    #bn = K.layers.BatchNormalization()(dense)
    #relu = K.layers.Activation('relu')(bn)
    #dropout = K.layers.Dropout(0.2)(relu)
    
    #dense = K.layers.Dense(64)(dropout)
    #bn = K.layers.BatchNormalization()(dense)
    #relu = K.layers.Activation('relu')(bn)
    #dropout = K.layers.Dropout(0.2)(relu)
    
    #dense = K.layers.Dense(32)(dropout)
    #bn = K.layers.BatchNormalization()(dense)
    #relu = K.layers.Activation('relu')(bn)
    #dropout = K.layers.Dropout(0.2)(relu)
    
    # Attention Layer
    attention_layer = cbam_spatial_attention(feature_input)
    
    # Classification layers
    dense = K.layers.Dense(256, activation='relu')(attention_layer)
    bn = K.layers.BatchNormalization()(dense)
    dropout = K.layers.Dropout(0.2)(bn)
    
    # Classification output layer
    outputs = K.layers.Dense(10, activation='softmax')(dropout)
    
    top_model = K.Model(inputs=feature_input, outputs=outputs)
    
    optimizer = K.optimizers.Adam(learning_rate=0.001)

    # Compile the top model
    top_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    # Train the top model
   
    history = top_model.fit(
    train_features, Y_train_p,
    validation_data=(test_features, Y_test_p),
    epochs=10, batch_size=32,
    callbacks=callbacks,
    verbose=2
    )

    # Get predictions on test set
    print("\nGenerating predictions and visualization...")
    test_predictions = top_model.predict(test_features)

    # Generate all visualizations
    plot_training_history(history)
    plot_confusion_matrix(Y_test_p, test_predictions)
    plot_class_accuracy(Y_test_p, test_predictions)
    print("\nVisualization files have been saved!")

    # After training top_layers, save the full model
    print("Creating and saving full model...")
    try:
        full_model = K.Sequential([key_model, top_model])
        full_model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        sample_input = X_train_p[:1]  # Sample input for model summary
        _ = full_model(sample_input)  # Build the model to print summary
        full_model.save('cifar10.h5')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")