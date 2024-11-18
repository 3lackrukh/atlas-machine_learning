#!/usr/bin/env python3
""" Module for training a CNN to classify CIFAR 10 dataset
    using a pre-built Keras Application for transfer learning """
from tensorflow import keras as K
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
        # Define available operations and their ranges
        self.operations = {
            'rotate': (0, 30),          # Degrees
            'translate_x': (-0.3, 0.3),  # Fraction of width
            'translate_y': (-0.3, 0.3),  # Fraction of height
            'brightness': (0.1, 1.9),    # Factor
            'contrast': (0.1, 1.9),      # Factor
            'zoom': (0.8, 1.2),          # Factor
        }
        # Initialize Keras image processing layers    
        # Using default behaviors unless we specifically want to change them
        self.rotate = K.layers.RandomRotation(1.0)  # Uses defaults
        self.translate = K.layers.RandomTranslation(1.0, 1.0)  # Uses defaults
        self.zoom = K.layers.RandomZoom(1.0)  # Uses defaults
        
        # If we want to change from defaults, we should be explicit:
        # self.rotate = K.layers.RandomRotation(
        #     1.0,
        #     fill_mode='nearest',     # Different from default 'reflect'
        #     interpolation='bicubic' # Different from default 'bilinear'
        # )
        
    def __call__(self, image):
        # Select random operation
        op_name = np.random.choice(list(self.operations.keys()))
        # Random magnitude between 0 and 1
        magnitude = np.random.random()
        
        # Scale magnitude to operation's range
        op_range = self.operations[op_name]
        scaled_magnitude = op_range[0] + (op_range[1] - op_range[0]) * magnitude
        
        # Apply the selected operation
        return self.apply_op(image, op_name, scaled_magnitude)
    
    def apply_op(self, image, op_name, magnitude):
        """Apply the selected operation with given magnitude"""
        if isinstance(image, np.ndarray):
            image = K.ops.convert_to_tensor(image)
        
        # Expand dimensions if needed
        if len(image.shape) == 3:
            image = K.ops.expand_dims(image, 0)
            
        if op_name == 'rotate':
            return self.rotate(image, [magnitude/30.0])  # Normalize to [-1,1]
        
        elif op_name == 'translate_x':
            return self.translate(image, [magnitude, 0])
            
        elif op_name == 'translate_y':
            return self.translate(image, [0, magnitude])
            
        elif op_name == 'brightness':
            return K.image.adjust_brightness(image, magnitude - 1.0)
            
        elif op_name == 'contrast':
            return K.image.adjust_contrast(image, magnitude)
            
        elif op_name == 'zoom':
            return self.zoom(image, [magnitude-1.0])
        
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
    """
    END TRIVIAL AUGMENTATION
    """
    
    """ CLASS SPECIFIC AUGMENTATION    
        # CIFAR-10 classes,
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
        
        #Define classes to augment
        augment_classes = ['cat', 'dog']
        
        # Increase images in problem classes by 10%
        images_to_add = int(5000 * 0.1)
        
        # Configure augmentation
        datagen = K.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Initialize lists to store balanced data
        processed_x = []
        processed_y = []
        
        # Dictionary to keep track of samples for each class
        class_samples = {cls: [] for cls in classes}
        
        for class_idx, class_name in enumerate(classes):
            # Get samples for this class
            class_mask = Y[:, 0] == class_idx
            X_class = X[class_mask]
        
                    
            # Add original samples
            processed_x.extend(X_class)
            processed_y.extend([class_idx] * len(X_class))
            
            # Add augmented samples only for target classes
            if class_name in augment_classes:
                print(f"\nAugmenting {class_name} class with {images_to_add} new images")
                
                progress = ProgressBar(
                    total=images_to_add,
                    prefix=f'Augmenting {class_name}'
                )
                
                # Generate exactly images_to_add new samples
                for i in range(images_to_add):
                    # Pick a random image from this class
                    idx = np.random.randint(len(X_class))
                    img = X_class[idx:idx+1]
                    
                    # Generate augmented version
                    aug_img = next(datagen.flow(
                        img, 
                        batch_size=1,
                        seed=np.random.randint(1000)  # Random seed for variation
                    ))[0]
                    
                    processed_x.append(aug_img)
                    processed_y.append(class_idx)
                    progress.update(1)
                
                progress.close()
                print(f"Final count for {class_name}: {len(X_class) + images_to_add}")
        
        # Convert to numpy arrays
        print('Converting training images to np.array')
        X_p = np.array(processed_x)
        Y_p = K.utils.to_categorical(processed_y, 10)
        
        # Print final distribution
        print("\nFinal class distribution:")
        for class_idx, class_name in enumerate(classes):
            count = np.sum(np.argmax(Y_p, axis=1) == class_idx)
            print(f"    {class_name}: {count} images")
        
    else: 
        Y_p = K.utils.to_categorical(Y, 10)
        X_p = np.array(X)
        
    # Rescale pixel values
    X_p = K.applications.efficientnet_v2.preprocess_input(X_p)
    
    return X_p, Y_p
    END CLASS SPECIFIC AUGMENTATION """
      

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

    """
    With l2 regularization and he_normal initialization
    mlp = K.Sequential([
        K.layers.Dense(1280 // 16, kernel_regularizer=K.regularizers.l2(0.005),
                        kernel_initializer='he_normal', activation='relu'),
        K.layers.Dense(1280, kernel_regularizer=K.regularizers.l2(0.005),
                        activation='sigmoid')
    ])
    """
    mlp = K.Sequential([
        K.layers.Dense(1280 // 16, activation='relu'),
        K.layers.Dense(1280, activation='sigmoid')
    ])
    
    channel_attention = mlp(avg_pool) + mlp(max_pool)
    x_chan = x_reshape * channel_attention
    
    avg_spatial = K.layers.Lambda(lambda x: K.backend.mean(x, axis=-1, keepdims=True))(x_chan)
    max_spatial = K.layers.Lambda(lambda x: K.backend.max(x, axis=-1, keepdims=True))(x_chan)
    spatial_concat = K.layers.Concatenate(axis=-1)([avg_spatial, max_spatial])
    
    
    spatial_attention = K.layers.Conv2D(1, 7, padding='same')(spatial_concat)
    spatial_attention = K.layers.BatchNormalization()(spatial_attention)
    spatial_attention = K.layers.Activation('sigmoid')(spatial_attention)
    
    x_spatial = x_chan * spatial_attention
    """
    ATTENTION BASED DROPOUT
    x_dropout = K.layers.Lambda(
        lambda x: attention_based_dropout(x[0], x[1], threshold=0.05, drop_rate=0.75)
        )([x_spatial, spatial_attention])
    """

    x_flat = K.layers.GlobalAveragePooling2D()(x_spatial)
    
    return x_flat


def attention_based_dropout(x, attention_weights, threshold=0.5, drop_rate=0.2):
    """
    Drops neurons in regions where attention_weights < threshold
    x: input features
    attention_weights: from CBAM
    threshold: attention value below which to apply dropout
    drop_rate: probability of dropping neurons in low-attention regions
    """
    # Create binary mask for low-attention regions
    low_attention_mask = K.backend.cast(attention_weights < threshold, 'float32')
    
    # Generate random dropout mask for those regions
    random_mask = K.backend.random_uniform(K.backend.shape(x)) > drop_rate
    random_mask = K.backend.cast(random_mask, 'float32')
    
    # Only apply dropout to low-attention regions
    dropout_mask = K.backend.ones_like(x) * (1 - low_attention_mask) + random_mask * low_attention_mask
    
    # Count dropped neurons
    total_neurons = K.backend.cast(K.backend.prod(K.backend.shape(dropout_mask)), 'float32')
    dropped_neurons = total_neurons - K.backend.sum(dropout_mask)
    dropout_rate = dropped_neurons / total_neurons
    
    return x * dropout_mask



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
    
     # Freeze the base model layers
    print('Freezing key_model layers')
    key_model.trainable = False
    
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
    print('Loading CIFAR-10 dataset ...')    
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()

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
    dropout = K.layers.Dropout(0.3)(attention_layer)

    
    # Classification layers
    """ Classification with l2 normalization
    dense = K.layers.Dense(256, kernel_initializer='he_normal',
                           kernel_regularizer=K.regularizers.l2(0.015),
                           activation='relu')(attention_layer)
    """
    dense = K.layers.Dense(256, activation='relu')(attention_layer)
    bn = K.layers.BatchNormalization()(dense)
    dropout = K.layers.Dropout(0.4)(bn)
    
    # Classification output layer
    outputs = K.layers.Dense(10, activation='softmax')(dropout)
    
    top_model = K.Model(inputs=feature_input, outputs=outputs)
    
    optimizer = K.optimizers.Adam(learning_rate=0.001)

    # Compile the top model
    top_model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    print("top_model compiled!")
    
    # Train the top model
    print('Training top_model on extracted features in batches')
    history = top_model.fit(
        batch_feature_data('train'),
        validation_data=batch_feature_data('test', batch_size=32),
        batch_size=32,
        epochs=10,
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