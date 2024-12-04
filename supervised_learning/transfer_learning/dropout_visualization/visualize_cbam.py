import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as K

def visualize_preprocessing(image, resized, features):
    """Show preprocessing steps"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Image Preprocessing Pipeline', size=16)
    
    axes[0].imshow(image[0])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(resized[0])
    axes[1].set_title('Resized Image (224x224)')
    axes[1].axis('off')
    
    feature_mean = np.mean(features[0], axis=-1)
    feature_mean = (feature_mean - feature_mean.min()) / (feature_mean.max() - feature_mean.min())
    axes[2].imshow(feature_mean, cmap='viridis')
    axes[2].set_title('EfficientNetV2 Features')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()  # Explicitly show first figure

def visualize_cbam_dropout(features):
    """
    Visualizes the CBAM spatial attention dropout process
    """
    plt.figure(figsize=(15, 15))  # Create new figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('CBAM Analysis', size=16)
    
    # 1. Feature Map Overview
    feature_mean = np.mean(features[0], axis=-1)
    feature_mean = (feature_mean - feature_mean.min()) / (feature_mean.max() - feature_mean.min())
    im0 = axes[0, 0].imshow(feature_mean, cmap='viridis')
    axes[0, 0].set_title('Feature Map (Average across channels)')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], label='Feature Intensity')
    
    # 2. Channel Attention
    channel_weights = np.mean(features[0], axis=(0, 1))
    axes[0, 1].bar(range(len(channel_weights)), channel_weights)
    axes[0, 1].set_title('Channel Attention Weights')
    axes[0, 1].set_xlabel('Channel Index')
    axes[0, 1].set_ylabel('Weight')
    
    # 3. Spatial Attention
    spatial_attention = np.mean(features[0], axis=-1)
    spatial_attention = (spatial_attention - spatial_attention.min()) / \
                       (spatial_attention.max() - spatial_attention.min())
    im2 = axes[1, 0].imshow(spatial_attention, cmap='jet')
    axes[1, 0].set_title('Spatial Attention Map')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Attention Strength')
    
    # 4. Dropout Mask
    dropout_threshold = 0.3
    dropout_rate = 0.75
    mask = np.random.rand(*spatial_attention.shape) > \
           (dropout_rate * (spatial_attention < dropout_threshold))
    im3 = axes[1, 1].imshow(mask, cmap='RdYlBu')
    axes[1, 1].set_title('Attention-Based Dropout Mask\n(Blue=Keep, Red=Drop)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], label='Keep/Drop')
    
    plt.tight_layout()
    plt.show()  # Explicitly show second figure

def run_visualization():
    try:
        # Load data and model
        (_, _), (X_test, Y_test) = K.datasets.cifar10.load_data()
        sample_image = X_test[0:1].astype('float32') / 255.0
        
        custom_objects = {
            'LinearDecay': LinearDecay,
            'cbam_spatial_attention': cbam_spatial_attention,
            'attention_based_dropout': attention_based_dropout
        }
        
        with K.utils.custom_object_scope(custom_objects):
            model = K.models.load_model('cifar10.h5')
        
        # Get features
        feature_extractor = model.get_layer('sequential')
        resized = K.layers.Resizing(224, 224, interpolation='bicubic')(sample_image)
        features = feature_extractor.predict(sample_image, verbose=0)
        
        # Show both visualization panels
        visualize_preprocessing(sample_image, resized, features)
        visualize_cbam_dropout(features)
        
        # Print classification results
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        true_class = class_names[Y_test[0][0]]
        prediction = model.predict(sample_image, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        
        print(f"\nTrue class: {true_class}")
        print(f"Predicted class: {predicted_class}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    run_visualization()