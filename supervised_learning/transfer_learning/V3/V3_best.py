V3 template

#!/usr/bin/env python3
""" Module for training a CNN to classify CIFAR 10 dataset
    using a pre-built Keras Application for transfer learning """
from tensorflow import keras as K

def preprocess_data(X, Y):
    """
    Preprocesses the data for training.

    Parameters:
        X (numpy.ndarray): Input data of shape (m, 32, 32, 3)
                           containing the CIFAR 10 images.
        Y (numpy.ndarray): Labels for the input data.

    Returns:
        X_p (numpy.ndarray): Preprocessed input data
        Y_p (numpy.ndarray): Preprocessed labels in one-hot encoded format
    """
    # Scale pixel values
    X_p = K.applications.efficientnet_v2.preprocess_input(X)
    # Convert labels to one-hot encoded format
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def extract_features_with_progress(model, X, dataset_type, h5file, dataset_name, 
                                 start_idx=0, batch_size=10):
    """Extract features with progress tracking"""
    
    for i in range(start_idx, X.shape[0], batch_size):
            end_idx = min(i + batch_size, X.shape[0])
            batch_features = model.predict(X[i:end_idx], verbose=0)
            h5file[dataset_name][i:end_idx] = batch_features

def batch_extract(key_model, X_train_p, X_test_p, Y_train_p, Y_test_p, shuffle_idx, batch_size=10):
    """Process features in batches with progress tracking"""
    try:
        with h5py.File('cifar10_features.h5', 'r') as f:
            print('Features file detected\n    validating...')
            last_train_idx = f['train_features'].shape[0]
            last_test_idx = f['test_features'].shape[0]
            f.close()
    except Exception as e:
        
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

def cbam_spatial_attention(x):
    """
    Implements CBAM spatial attention mechanism.
    
    Parameters:
        x: Input tensor
    
    Returns:
        Tensor with spatial attention applied
    """
    x_reshape = K.layers.Reshape((12, 12, 1280))(x)
    
    # Channel attention
    avg_pool = K.layers.GlobalAveragePooling2D(keepdims=True)(x_reshape)
    max_pool = K.layers.GlobalMaxPooling2D(keepdims=True)(x_reshape)

    mlp = K.Sequential([
        K.layers.Dense(1280 // 16, activation='relu'),
        K.layers.Dense(1280, activation='sigmoid')
    ])
    
    channel_attention = mlp(avg_pool) + mlp(max_pool)
    x_chan = x_reshape * channel_attention
    
    # Spatial attention
    avg_spatial = K.layers.Lambda(lambda x: K.backend.mean(x, axis=-1, keepdims=True))(x_chan)
    max_spatial = K.layers.Lambda(lambda x: K.backend.max(x, axis=-1, keepdims=True))(x_chan)
    spatial_concat = K.layers.Concatenate(axis=-1)([avg_spatial, max_spatial])
    
    spatial_attention = K.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
    x_spatial = x_chan * spatial_attention
    
    # Apply attention-based dropout
    x_dropout = K.layers.Lambda(
        lambda x: attention_based_dropout(x[0], x[1], threshold=0.05, drop_rate=0.75)
    )([x_spatial, spatial_attention])
    
    return K.layers.GlobalAveragePooling2D()(x_dropout)

if __name__ == '__main__':
    # Initialize model components
    print('Initializing Resizer input layer: bicubic interpolation')
    Resizer = K.layers.Resizing(384, 384, interpolation='bicubic')

    print('Initializing base_model: EfficientNetV2S