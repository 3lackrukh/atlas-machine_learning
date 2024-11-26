#!/usr/bin/env python3
""" Module for training a CNN to classify CIFAR 10 dataset
    using a pre-built Keras Application for transfer learning """
from tensorflow import keras as K
import tensorflow as tf
import h5py
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ProgressBar:
    """Custom progress bar implementation"""
    def __init__(self, total, prefix='Progress:', length=50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        
        # Initialize progress bar
        self._print_progress(0)
    
    def update(self, amount):
        """Update progress bar"""
        self.current += amount
        self._print_progress(self.current)
    
    def _print_progress(self, current):
        """Print the progress bar"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate progress metrics
        progress = min(1.0, current / self.total)
        filled_length = int(self.length * progress)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        
        # Calculate speed and ETA
        speed = current / elapsed if elapsed > 0 else 0
        remaining = (self.total - current) / speed if speed > 0 else 0
        
        # Format time strings
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(remaining)
        
        # Create progress line
        line = (f'\r{self.prefix} |{bar}| '
                f'{current}/{self.total} '
                f'({progress:.1%}) '
                f'[time elapsed:{elapsed_str}<remaining:{eta_str}>, '
                f'{speed:.1f} samples/s]')
        
        # Clear the entire line before printing new progress
        sys.stdout.write('\033[K')  # Clear line from cursor to end
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Print newline if complete
        if current >= self.total:
            print()
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds into human-readable string"""
        return str(timedelta(seconds=int(seconds)))
    
    def close(self):
        """Finish progress bar"""
        if self.current < self.total:
            self._print_progress(self.total)
        print()

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

class TrivialAugment:
    def __init__(self):
        # Initialize Keras preprocessing layers with conservative ranges
        self.operations = [
            'rotate', 'translate_x', 'translate_y', 'brightness', 
            'contrast', 'zoom', 'flip'
        ]
        # All K.layers defaults:
        #  fill_mode='reflect'
        #       options: constant, nearest, wrap
        #  interpolation='bilinear'
        #       options: nearest, bicubic
        
        # Initialize layers with specific ranges
        self.rotate = K.layers.RandomRotation(20/360)  # 30 degrees
        self.translate = K.layers.RandomTranslation(0.1, 0.1)  # 30% movement
        self.brightness = K.layers.RandomBrightness(0.9)  # Factor of 0.1 to 1.9
        self.contrast = K.layers.RandomContrast(0.9)     # Factor of 0.1 to 1.9
        # self.zoom = K.layers.RandomZoom(0.2)            # 0.8 to 1.2 zoom
        # self.shear = K.layers.RandomShear(0.3)          # 30% shear
        # self.sharpness = K.layers.RandomSharpness(0.95) # 0.1 to 2.0 range
        self.flip = K.layers.RandomFlip('horizontal')
        
    def __call__(self, image):
        # Select random operation
        op_name = np.random.choice(self.operations)
        
        # Ensure image is tensor
        if isinstance(image, np.ndarray):
            image = K.utils.img_to_array(image)
        
        # Apply the selected operation
        if op_name == 'rotate':
            return self.rotate(image, training=True)
        elif op_name == 'translate_x':
            return self.translate(image, training=True)
        elif op_name == 'translate_y':
            return self.translate(image, training=True)
        elif op_name == 'brightness':
            return self.brightness(image, training=True)
        elif op_name == 'contrast':
            return self.contrast(image, training=True)
        #elif op_name == 'zoom':
            #return self.zoom(image, training=True)
        # elif op_name == 'shear':
            # return self.shear(image, training=True)
        # elif op_name == 'sharpness':
            # return self.sharpness(image, training=True)
        elif op_name == 'flip':
            return self.flip(image, training=True)
        
        return image


def preprocess_data(X, Y, training_set=False):
    """
    Preprocesses the data for training.

    Parameters:
        X (numpy.ndarray): Input data of shape (m, 32, 32, 3)
                           containing the CIFAR 10 images.
        Y (numpy.ndarray): Labels for the input data.
        training (bool): Indicates whether the data is for training or not.

    Returns:
        X_p (numpy.ndarray): Preprocessed input data
        Y_p (numpy.ndarray): Preprocessed labels in one-hot encoded format
    """
    if training_set:
        
        """ TRIVIAL AUGMENTATION IMPLEMENTATIONS """  
        augmenter = TrivialAugment()
        processed_x = []
        
        print("Applying TrivialAugment to training data...")
        progress = ProgressBar(total=len(X), prefix='Augmenting')
        
        for i in range(len(X)):
            img = X[i]
            # Apply augmentation
            aug_img = augmenter(img)
            processed_x.append(aug_img)
            progress.update(1)
            
        X_p = np.array(processed_x)
        Y_p = K.utils.to_categorical(Y, 10)
    else:
        X_p = np.array(X)
        Y_p = K.utils.to_categorical(Y, 10)
    
    # Apply EfficientNetV2 preprocessing
    X_p = K.applications.efficientnet_v2.preprocess_input(X_p)
    
    return X_p, Y_p
      

def extract_features_with_progress(model, X, dataset_type, h5file, dataset_name,
                                 start_idx=0, batch_size=10):
    """Extract features with progress tracking"""
    total_samples = X.shape[0] - start_idx
    
    # Create progress bar
    progress = ProgressBar(
        total=total_samples,
        prefix=f'{dataset_type} Feature Extraction:'
    )
    
    try:
        for i in range(start_idx, X.shape[0], batch_size):
            end_idx = min(i + batch_size, X.shape[0])
            batch_features = model.predict(X[i:end_idx], verbose=0)
            h5file[dataset_name][i:end_idx] = batch_features
            
            # Update progress
            progress.update(end_idx - i)
    finally:
        progress.close()


def batch_extract(key_model, X_train_p, X_test_p, Y_train_p, Y_test_p, shuffle_idx, batch_size=10):
    """Process features in batches with progress tracking"""
    if os.path.exists('cifar10_features.h5'):
        with h5py.File('cifar10_features.h5', 'r') as f:
            last_train_idx = f['train_features'].shape[0]
            last_test_idx = f['test_features'].shape[0]
        
        missing_train = X_train_p.shape[0] - last_train_idx
        missing_test = X_test_p.shape[0] - last_test_idx
        
        if missing_train > 0 or missing_test > 0:
            print(f'Found {missing_train} missing training features')
            print(f'Found {missing_test} missing test features')
            
            with h5py.File('cifar10_features.h5', 'a') as f:
                if missing_train > 0:
                    extract_features_with_progress(
                        key_model, X_train_p, 'Training', f, 'train_features',
                        start_idx=last_train_idx, batch_size=batch_size
                    )
                
                if missing_test > 0:
                    extract_features_with_progress(
                        key_model, X_test_p, 'Testing', f, 'test_features',
                        start_idx=last_test_idx, batch_size=batch_size
                    )
    else:
        print('No saved features detected. Creating new feature file...')
        with h5py.File('cifar10_features.h5', 'w') as f:
            # Create datasets
            f.create_dataset('train_features', 
                           shape=(X_train_p.shape[0], 12, 12, 1280), 
                           dtype='float32')
            f.create_dataset('test_features', 
                           shape=(X_test_p.shape[0], 12, 12, 1280), 
                           dtype='float32')
            f.create_dataset('train_labels', data=Y_train_p)
            f.create_dataset('test_labels', data=Y_test_p)
            f.create_dataset('shuffle_idx', data=shuffle_idx)
            
            # Extract features
            extract_features_with_progress(
                key_model, X_train_p, 'Training', f, 'train_features',
                batch_size=batch_size
            )
            extract_features_with_progress(
                key_model, X_test_p, 'Testing', f, 'test_features',
                batch_size=batch_size
            )

def batch_feature_data(collection, features_file='cifar10_features.h5',
                       batch_size=32, predict=False):
    """
    Divides features saved to disk into batches for training or prediction

    parameters:
        collection: string name of collection to train from
        features_file: string filepath for saved features
                       Defaults to 'cifar10_features.h5'
        batch_size: integer batch_size for training
                    Defaults to 32
        predict: boolean for prediction
                 Defaults to False
    
    Yields:
        tuple of (features, labels) for each batch
        or features only during prediction
    """
    with h5py.File(features_file, 'r') as f:        
        samples = f[f'{collection}_features'].shape[0]
        
        while True:
            for i in range(0, samples, batch_size):
                end_idx = min(i + batch_size, samples)
                if predict:
                    yield f[f'{collection}_features'][i:end_idx]
                else:
                    yield (f[f'{collection}_features'][i:end_idx],
                           f[f'{collection}_labels'][i:end_idx])


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
    
    avg_spatial = K.layers.Lambda(lambda x: K.backend.mean(x, axis=-1, keepdims=True))(x_chan)
    max_spatial = K.layers.Lambda(lambda x: K.backend.max(x, axis=-1, keepdims=True))(x_chan)
    spatial_concat = K.layers.Concatenate(axis=-1)([avg_spatial, max_spatial])
    
    
    spatial_attention = K.layers.Conv2D(1, 12, padding='same')(spatial_concat)
    spatial_attention = K.layers.Activation('sigmoid')(spatial_attention)
    x_spatial = x_reshape * spatial_attention
    
    x_s_drop = attention_based_dropout(x_spatial, spatial_attention, low_threshold=0.3, drop_rate=0.75)
    x_attn = x_chan + x_s_drop
    
    return x_attn


def attention_based_dropout(x, attention_weights, low_threshold=0.2, drop_rate=0.2):
    """
    Drops neurons in regions where attention_weights < threshold
    x: input features
    attention_weights: from CBAM
    threshold: attention value below which to apply dropout
    drop_rate: probability of dropping neurons in low-attention regions
    """
    # Ensure drop_rate is broadcast to match feature dimensions
    if isinstance(drop_rate, (int, float)):   # scalar
        drop_rate = K.backend.cast(
            K.backend.ones_like(x) * drop_rate,
            dtype='float32'
        )
    else: # tensor
        drop_rate = K.backend.reshape(drop_rate, (-1, 1, 1, 1)) * K.backend.ones_like(x)
    
    # Create binary mask for low-attention regions
    attention_mask = K.backend.cast(
        attention_weights < low_threshold, 'float32')
    
    # Generate random dropout mask for those regions
    random_mask = K.backend.cast(
        K.backend.random_uniform(K.backend.shape(x)) > drop_rate,
        'float32'
    )
    
    # Only apply dropout to low-attention regions
    dropout_mask = K.backend.ones_like(x) * (1 - attention_mask) + random_mask * attention_mask
    
    return x * dropout_mask


class LinearDecay(K.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, final_lr, decay_steps):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps

    def __call__(self, step):
        lr = self.initial_lr - ((self.initial_lr - self.final_lr) / K.backend.cast(self.decay_steps, 'float32')) * K.backend.cast(step, 'float32')
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "final_lr": self.final_lr,
            "decay_steps": self.decay_steps
        }



if __name__ == '__main__':
     # Resize layer for input images
    # Attempt 1 using bicubic interpolation
    # Chosen for balanced detail preservation vs speed
    print('Initializing Resizer input layer: bicubic interpolation')
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
    print('Compiling Resizer and base_model into key_model ...')
    key_model = K.Sequential([
        Resizer,
        base_model
    ])
    # Build the model
    key_model.build(input_shape=(None, 384, 384, 3))
     # Freeze the base model layers except 9
    print('Freezing key_model layers')
    key_model.trainable = True
    for layer in key_model.layers[:-9]:
        layer.trainable = False
    
    print('Trainable Layers:')
    for layer in key_model.layers:
        if layer.trainable:
            print(f'{layer.name} : {layer.count_params()} parameters')
   
    # Load and pre-process CIFAR-10 dataset
    print('Loading CIFAR-10 dataset ...')    
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

 # Add callbacks for early stopping and model checkpoint
    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ModelCheckpoint(
            'cifar10.h5',
            monitor='val_accuracy',
            save_best_only=True,
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=2,
            min_lr=0.00001,
        )
    ]
    
    print('  Preprocessing training data ...')
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train, training_set=True)
    print('  Preprocessing testing data ...')
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)
    
    # Check for saved features
    print('Checking for saved features...')
    if os.path.exists('cifar10_features.h5'):

        print('Saved features detected. Validating ...')
        print('  Indexing training images to match features order')
        with h5py.File('cifar10_features.h5', 'r') as f:
            shuffle_idx = f['shuffle_idx']
            X_train_p = X_train_p[shuffle_idx]
            Y_train_p = Y_train_p[shuffle_idx]
            f.close()
    
    else:
        print('No features detected...')
        
        # Shuffle training data
        print('  Shuffling training data...')
        # Create shuffled index to maintain matching shuffle order
        shuffle_idx = np.random.permutation(len(X_train_p))
        # Shuffle training images and labels 

        X_train_p = X_train_p[shuffle_idx]
        Y_train_p = Y_train_p[shuffle_idx]
        
        print('  Extracting features ...')
        
    print(f'training data is of type: {type(X_train_p)}')
    print(f'trainig data shape: {(X_train_p.shape)}')    
    batch_extract(key_model, X_train_p, X_test_p, Y_train_p, Y_test_p, batch_size=10, shuffle_idx=shuffle_idx)
   
    # Create new top model
    print('Creating and training top_model ...')
    # Attempt 1 categorical softmax output layer to establish a baseline
    # Attempt 2 attention and classification layers
    feature_input=K.Input(shape=(12, 12, 1280,)) # Shape of EfficientNetV2S output
    
    
    # Attention Layer
    attention_layer = cbam_spatial_attention(feature_input)

    #dropout = K.layers.Dropout(0.3)(attention_layer)
    
    dense = K.layers.Dense(256, activation='relu',
                           kernel_initializer='he_normal',
                           bias_initializer='zeros',
                           )(attention_layer)
    bn = K.layers.BatchNormalization()(dense)
    dropout = K.layers.Dropout(0.2)(bn)
    pool = K.layers.GlobalAveragePooling2D()(dropout)
    
    # Classification output layer
    outputs = K.layers.Dense(10, activation='softmax')(pool)
    
    top_model = K.Model(inputs=feature_input, outputs=outputs)
    
    # Configure optimizer
    initial_lr = 0.001
    final_lr = 0.00039
    decay_steps = len(X_train) // 32 * 12  # Decay learning rate every batch for 10 epochs
   
    lr_schedule = LinearDecay(initial_lr, final_lr, decay_steps)
    optimizer = K.optimizers.Adam(learning_rate=lr_schedule)

    top_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("top_model compiled!")
    
    # Train the top model
    print('Training top_model on extracted features in batches')
    history = top_model.fit(
        batch_feature_data('train'),
        validation_data=batch_feature_data('test', batch_size=32),
        batch_size=32,
        epochs=12,
        steps_per_epoch=X_train_p.shape[0] // 32,
        validation_steps=X_test_p.shape[0] // 32,
        callbacks=callbacks,
        verbose=1
    )


    # Get predictions on test set
    print("\nGenerating predictions and visualization...")
    test_predictions = top_model.predict(
        batch_feature_data('test', predict=True),
        steps=X_test_p.shape[0] // 32 + 1,
        verbose=1
        
    )

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