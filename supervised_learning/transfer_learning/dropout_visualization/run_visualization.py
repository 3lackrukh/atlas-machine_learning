import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
from visualize_cbam import visualize_cbam_dropout

# Import the custom classes from your original file
from transfer_learning import (LinearDecay, cbam_spatial_attention, 
                             attention_based_dropout, preprocess_data)

def visualize_full_pipeline(image, model, true_class=None):
    """
    Visualize the complete pipeline including preprocessing and CBAM attention
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image[0])  # First image in batch
    plt.title("Original Image")
    plt.axis('off')
    
    # Get the sequential layer (which contains resizer and EfficientNetV2)
    feature_extractor = model.get_layer('sequential')
    
    # Get resized image
    resized = K.layers.Resizing(224, 224, interpolation='bicubic')(image)
    plt.subplot(1, 3, 2)
    plt.imshow(resized[0])
    plt.title("Resized Image (224x224)")
    plt.axis('off')
    
    # Get features using the full feature extractor
    features = feature_extractor(image)
    
    # Feature visualization
    feature_viz = np.mean(features[0], axis=-1)
    plt.subplot(1, 3, 3)
    plt.imshow(feature_viz, cmap='viridis')
    plt.title("EfficientNetV2 Features")
    plt.axis('off')
    
    if true_class is not None:
        plt.suptitle(f"Class: {true_class}")
    
    plt.tight_layout()
    plt.show()
    
    # Now visualize the CBAM attention
    return visualize_cbam_dropout(features)

def run_visualization():
    try:
        # Create custom objects dictionary
        custom_objects = {
            'LinearDecay': LinearDecay,
            'cbam_spatial_attention': cbam_spatial_attention,
            'attention_based_dropout': attention_based_dropout
        }
        
        # Load the model with custom objects
        with K.utils.custom_object_scope(custom_objects):
            full_model = K.models.load_model('cifar10.h5')
        
        # Load and prepare a sample image
        (_, _), (X_test, Y_test) = K.datasets.cifar10.load_data()
        sample_image = X_test[0:1].astype('float32') / 255.0
        
        # Get class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        true_class = class_names[Y_test[0][0]]
        
        # Visualize the full pipeline
        visualize_full_pipeline(sample_image, full_model, true_class)
        
        # Get prediction
        prediction = full_model.predict(sample_image, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        
        print(f"\nTrue class: {true_class}")
        print(f"Predicted class: {predicted_class}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all files are in the same directory and the model is properly saved")
        raise  # This will show the full error trace

if __name__ == "__main__":
    run_visualization()