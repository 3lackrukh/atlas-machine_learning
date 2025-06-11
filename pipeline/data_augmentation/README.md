# Data Augmentation

This project implements various data augmentation techniques for deep learning, focusing on image transformations using TensorFlow to enhance training datasets with less data.

## Description

Implementation of key data augmentation methods including image flipping, cropping, rotation, contrast adjustment, brightness modification, and hue changes using TensorFlow and the Stanford Dogs dataset from TensorFlow Datasets.

## Requirements

- Python 3.9
- Ubuntu 20.04 LTS
- numpy (version 1.25.2)
- tensorflow (version 2.15)
- tensorflow-datasets (version 4.9.2)
- pycodestyle (version 2.11.1)

## Installation

Install TensorFlow Datasets:
```bash
pip install --user tensorflow-datasets==4.9.2
```

## Files

- 0-flip.py - Horizontal image flipping implementation
- 1-crop.py - Random image cropping implementation
- 2-rotate.py - 90-degree counter-clockwise rotation implementation
- 3-contrast.py - Random contrast adjustment implementation
- 4-brightness.py - Random brightness modification implementation
- 5-hue.py - Hue change implementation

## Augmentation Techniques Implemented

### Image Flipping
Performs horizontal flipping of images to increase dataset diversity and improve model generalization.

### Random Cropping
Extracts random crops from images to focus on different regions and create spatial variations.

### Image Rotation
Rotates images by 90 degrees counter-clockwise to simulate different orientations.

### Contrast Adjustment
Randomly adjusts image contrast within specified bounds to simulate different lighting conditions.

### Brightness Modification
Changes image brightness randomly to improve robustness to varying illumination.

### Hue Changes
Modifies the hue component of images to create color variations while preserving structure.

## Usage

Each augmentation technique can be tested using its corresponding main file:
- ./0-main.py (Test image flipping)
- ./1-main.py (Test random cropping)
- ./2-main.py (Test image rotation)
- ./3-main.py (Test contrast adjustment)
- ./4-main.py (Test brightness modification)
- ./5-main.py (Test hue changes)

## Learning Objectives

- Understanding data augmentation principles and benefits
- When and how to apply data augmentation techniques
- Various methods for performing image augmentation
- Using TensorFlow for automated data augmentation
- Improving model performance with limited training data

## Dataset

All augmentation techniques are demonstrated using the Stanford Dogs dataset from TensorFlow Datasets, which contains images of various dog breeds for classification tasks.
