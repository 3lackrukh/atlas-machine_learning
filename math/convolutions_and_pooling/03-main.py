#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
convolve_grayscale_valid = __import__('0-convolve_grayscale_valid').convolve_grayscale_valid
convolve_grayscale = __import__('3-convolve_grayscale').convolve_grayscale


if __name__ == '__main__':
    # Load dataset
    dataset = np.load('../../supervised_learning/data/MNIST.npz')
    images = dataset['X_train']
    print("Original shape:", images.shape)
    
    # Define kernel
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    
    # Apply both convolutions
    images_conv_valid = convolve_grayscale_valid(images, kernel)
    images_conv = convolve_grayscale(images, kernel, padding='valid', stride=(1, 1))
    
    print("Valid convolution shape:", images_conv_valid.shape)
    print("General convolution shape:", images_conv.shape)
    
    # Compare outputs
    if images_conv_valid.shape == images_conv.shape:
        difference = np.abs(images_conv_valid - images_conv)
        print("Max difference between outputs:", np.max(difference))
        print("Mean difference between outputs:", np.mean(difference))
    else:
        print("Outputs have different shapes!")

    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(images[0], cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title('Valid Convolution')
    plt.imshow(images_conv_valid[0], cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title('General Convolution')
    plt.imshow(images_conv[0], cmap='gray')
    
    plt.show()

    # If shapes match, show difference
    if images_conv_valid.shape == images_conv.shape:
        plt.title('Difference between outputs')
        plt.imshow(difference[0], cmap='hot')
        plt.colorbar()
        plt.show()