#!/usr/bin/env python3
""" module containing function that uses transfer learning to construct a
model that identifies images in the CIFAR-10 dataset """
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# After imports, before visualization functions:
def preprocess_data(X, Y):
    """ preprocesses images and labels for EfficientNetV2 model """
    X_p = K.applications.efficientnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

def save_features_batch(batch_features, batch_labels, phase, batch_num, total_samples):
    """Safely saves feature/label batches with validation"""
    feature_file = f'cifar10_{phase}_features.npy'
    label_file = f'cifar10_{phase}_labels.npy'
    
    try:
        if batch_num == 0:  # First batch - create files with metadata
            np.savez(f'cifar10_{phase}_metadata.npz',
                    feature_shape=batch_features.shape[1:],
                    total_samples=total_samples)
            np.save(feature_file, batch_features)
            np.save(label_file, batch_labels)
        else:  # Append mode for subsequent batches
            with open(feature_file, 'ab') as f:
                np.save(f, batch_features)
            with open(label_file, 'ab') as f:
                np.save(f, batch_labels)
        return True
    except Exception as e:
        print(f"Error saving batch {batch_num}: {str(e)}")
        return False

class MemoryEfficientGenerator:
    """Generator that yields batches directly without saving to disk"""
    def __init__(self, X, Y, model, phase, batch_size=25):
        self.X = X
        self.Y = Y
        self.model = model
        self.batch_size = batch_size
        self.steps = len(X) // batch_size
        self.phase = phase
        
        # No augmentation, just basic ImageDataGenerator
        if phase == 'train':
            self.datagen = K.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                horizontal_flip=True,
                fill_mode='reflect'
            )
            self.batch_generator = self.datagen.flow(X, Y, batch_size=batch_size, shuffle=True)
        else:
            self.datagen = K.preprocessing.image.ImageDataGenerator()
            
            self.batch_generator = self.datagen.flow(X, Y, batch_size=batch_size, shuffle=False)
    
    def __len__(self):
        return self.steps
        
    def flow(self):
        first_batch = True
        while True:
            batch_x, batch_y = next(self.batch_generator)
            if first_batch:
                print(f"\nDEBUG - First 5 raw samples:")
                for i in range(5):
                    print(f"Image shape: {batch_x[i].shape}, Label: {batch_y[i]}")
                
                batch_x_p, batch_y_p = preprocess_data(batch_x, batch_y)
                if first_batch:
                    print(f"\nDEBUG - First 5 preprocessed samples:")
                    for i in range(5):
                        print(f"Image shape: {batch_x_p[i].shape}, One-hot label: {batch_y_p[i]}, Class: {np.argmax(batch_y_p[i])}")
                
                batch_features = self.model.predict(batch_x_p, verbose=0)
                if first_batch:
                    print(f"\nDEBUG - Feature shape: {batch_features.shape}")
                    print(f"DEBUG - Feature values range: min={np.min(batch_features)}, max={np.max(batch_features)}")
                    first_batch = False
            else:
                batch_x_p, batch_y_p = preprocess_data(batch_x, batch_y)
                batch_features = self.model.predict(batch_x_p, verbose=0)
            yield batch_features, batch_y_p
            
class Sharpener(K.layers.Layer):
    def __init__(self):
        super().__init__()
        # Define kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]], dtype='float32')
        kernel = kernel[:, :, None, None] * np.ones((3, 3, 3, 3), dtype='float32')
        # Create kernel as a non-trainable weight
        self.kernel = self.add_weight(
            shape=(3, 3, 3, 3),
            initializer=K.initializers.Constant(kernel),
            trainable=False,
            name='sharpen_kernel'
        )

    def call(self, inputs):
        return K.backend.clip(
            K.backend.conv2d(inputs, self.kernel, padding='same'),
            -1, 1)

def extract_features(X, Y, model, phase, batch_size=25):
    """Extract and save features/labels in batches with progress tracking"""
    steps = len(X) // batch_size + (1 if len(X) % batch_size != 0 else 0)
    total_samples = len(X)
    
    print(f"\nExtracting features for {phase} set:")
    print(f"Total batches: {steps}")
    
    if phase == 'train':
        datagen = K.preprocessing.image.ImageDataGenerator(
            rotation_range=30, 
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            fill_mode='reflect'

        )
    else:
        datagen = K.preprocessing.image.ImageDataGenerator()
        
    batch_generator = datagen.flow(X, Y, batch_size=batch_size)
    
    for i in range(steps):
        if i % 10 == 0:  # Progress update every 10 batches
            print(f"Processing batch {i}/{steps}")
            
        batch_x, batch_y = next(batch_generator)
        batch_x_p, batch_y_p = preprocess_data(batch_x, batch_y)
        batch_features = model.predict(batch_x_p, verbose=0)
        
        if not save_features_batch(batch_features, batch_y_p, phase, i, total_samples):
            raise Exception(f"Failed to save batch {i}")
            
    print(f"Feature extraction completed for {phase} set")
    return f'cifar10_{phase}_features.npy', f'cifar10_{phase}_labels.npy'

def cleanup_feature_files(phase):
    """Cleanup temporary files after training"""
    try:
        import os
        files = [
            f'cifar10_{phase}_features.npy',
            f'cifar10_{phase}_labels.npy',
            f'cifar10_{phase}_metadata.npz'
        ]
        for file in files:
            if os.path.exists(file):
                os.remove(file)
        return True
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return False
    
def cbam_spatial_attention(x):
    """
    Implements CBAM spatial attention mechanism with two-stage dropout:
    1. Initial dropout to filter strong features
    2. Class-aware dropout to emphasize class-specific features
    """
    print(f"CBAM input shape: {x.shape}")
    
    x_reshape = K.layers.Reshape((7, 7, 1280))(x)
    print(f"After reshape: {x_reshape.shape}")
    
    # Channel attention
    avg_pool = K.layers.GlobalAveragePooling2D(keepdims=True)(x_reshape)
    max_pool = K.layers.GlobalMaxPooling2D(keepdims=True)(x_reshape)
    print(f"Pool shapes - avg: {avg_pool.shape}, max: {max_pool.shape}")
    
    mlp = K.Sequential([
        K.layers.Dense(1280 // 16, activation='relu'),
        K.layers.Dense(1280, activation='sigmoid')
    ])
    
    channel_attention = mlp(avg_pool) + mlp(max_pool)
    print(f"Channel attention shape: {channel_attention.shape}")
    x_chan = x_reshape * channel_attention
    print(f"After channel attention multiply: {x_chan.shape}")
    
    # Spatial attention
    avg_spatial = K.layers.Lambda(
        lambda x: K.backend.mean(x, axis=-1, keepdims=True)
        )(x_chan)
    max_spatial = K.layers.Lambda(
        lambda x: K.backend.max(x, axis=-1, keepdims=True)
        )(x_chan)
    print(f"Spatial attention shapes - avg: {avg_spatial.shape}, max: {max_spatial.shape}")

    spatial_concat = K.layers.Concatenate(axis=-1)([avg_spatial, max_spatial])
    print(f"After spatial concat: {spatial_concat.shape}")
    
    spatial_attention = K.layers.Conv2D(1, 7, padding='same')(spatial_concat)
    spatial_attention = K.layers.Activation('sigmoid')(spatial_attention)
    print(f"Spatial attention shape after Conv2D(1, 7): {spatial_attention.shape}")
    #spatial_attention = K.layers.Conv2D(1, 3, padding='same')(spatial_attention)
    #print(f"Spatial attention shape after Conv2D(1, 3): {spatial_attention.shape}")
    
    # Apply aggressive spatially aware dropout
    x_spatial = x_reshape * spatial_attention
    print(f"x_spatial shape after spatial attention multiply: {x_spatial.shape}")
    
    x_s_drop = attention_based_dropout(x_spatial, spatial_attention, low_threshold=0.3, drop_rate=0.75)
    x_attn = x_chan + x_s_drop
    print(f"x_attn shape after addition: {x_attn.shape}")
    
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

#Visualization functions:
def plot_training_history(history):
    """
    Plots training history including accuracy and loss curves.
    Saves plots as training_history.png
    """
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
    Plots confusion matrix and saves as confusion_matrix.png
    Also prints classification report
    """
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
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
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes,
                              target_names=class_names))

def plot_class_accuracy(y_true, y_pred):
    """
    Plots per-class accuracy and saves as class_accuracy.png
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
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.close()
    
    
if __name__ == "__main__":
    # Load and prepare CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Set up callback functions for training
    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ModelCheckpoint(
            'cifar10_top.h5',
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
    
    # STAGE 1: Define feature extraction model
    resizer = K.layers.Resizing(224, 224, interpolation='bicubic')
    #sharpener = Sharpener()
    base_model = K.applications.EfficientNetV2S(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='none',
        include_preprocessing=True
    )
    key_model = K.Sequential([resizer, base_model])

    # Build the model
    key_model.build(input_shape=(None, 32, 32, 3))
    # Leave 3 MBConv blocks unfrozen
    key_model.trainable = True
    for layer in key_model.layers[:-9]:
        layer.trainable = False
    
    print('Trainable Layers:')
    for layer in key_model.layers:
        if layer.trainable:
            print(f'{layer.name} : {layer.count_params()} parameters')
     
    # STAGE 2: Define and train classification model
    feature_inputs = K.Input(shape=(7, 7, 1280))
    cbam = cbam_spatial_attention(feature_inputs)
     # First class attention branch for dropout rate
    dense = K.layers.Dense(256, activation='relu',
                           kernel_initializer='he_normal',
                           bias_initializer='zeros',
                           )(cbam)
    bn = K.layers.BatchNormalization()(dense)
    dropout = K.layers.Dropout(0.2)(bn)
    #dense = K.layers.Dense(128, activation='relu',
    #                       kernel_initializer='he_normal',
    #                       bias_initializer='zeros',
    #                       kernel_regularizer=K.regularizers.l2(0.0001),
    #                       )(dropout)
    #bn = K.layers.BatchNormalization()(dense)
    #dropout = K.layers.Dropout(0.2)(bn)
    pooled = K.layers.GlobalAveragePooling2D()(dropout)
    print(f"shape after first Dense(256): {pooled.shape}")
   
    class_probs = K.layers.Dense(10, activation='softmax')(pooled)
    print(f"Final output class_probs shape: {class_probs.shape}")
   
    top_model = K.Model(inputs=feature_inputs, outputs=class_probs)
    
    # Configure optimizer
    initial_lr = 0.001
    final_lr = 0.00039
    decay_steps = len(X_train) // 25 * 10  # Decay learning rate every batch for 10 epochs
   
    lr_schedule = LinearDecay(initial_lr, final_lr, decay_steps)
    optimizer = K.optimizers.Adam(learning_rate=lr_schedule)

    top_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Extract features 
    # Create data generators
    print("\nInitializing data generators...")
    train_generator = MemoryEfficientGenerator(X_train, Y_train, key_model, 'train', batch_size=25)
    val_generator = MemoryEfficientGenerator(X_test, Y_test, key_model, 'test', batch_size=25)

    try:
        # Train the model
        print("\nStarting training...")
        history = top_model.fit(
            train_generator.flow(),
            validation_data=val_generator.flow(),
            steps_per_epoch=len(X_train) // 25,
            validation_steps=len(X_test) // 25,
            epochs=10,
            callbacks=callbacks,
            verbose=1,
            shuffle=False
        )
        
        # Get predictions for visualization
        print("\nGenerating predictions for visualization...")
        test_predictions = []
        test_labels = []
        test_generator = MemoryEfficientGenerator(X_test, Y_test, key_model, 'test', batch_size=25)
        
        steps = len(X_test) // 25
        for _ in range(steps):
            features, labels = next(test_generator.flow())
            predictions = top_model.predict(features, verbose=1)
            test_predictions.append(predictions)
            test_labels.append(labels)
        
        test_predictions = np.vstack(test_predictions)
        test_labels = np.vstack(test_labels)
        
        # Generate visualizations
        plot_training_history(history)
        plot_confusion_matrix(test_labels, test_predictions)
        plot_class_accuracy(test_labels, test_predictions)
        
        # Save model
        print("\nSaving full model...")
        top_model.load_weights('cifar10_top.h5')
        full_model = K.Sequential([key_model, top_model])
        full_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        full_model.save('cifar10.h5')
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during execution: {str(e)}")