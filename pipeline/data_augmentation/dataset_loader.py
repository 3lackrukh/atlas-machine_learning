#!/usr/bin/env python3
"""Shared dataset loader for memory-efficient testing"""

import tensorflow as tf
import tensorflow_datasets as tfds

# Global variable to cache the dataset and test images per seed
_cached_dataset = None
_test_images = {}  # Cache images by seed


def get_test_image(seed=0):
    """
    Get a single test image from Stanford Dogs dataset for a specific seed.
    Caches the dataset to avoid multiple loads and caches images per seed.

    Args:
        seed: Random seed to use for image selection (default: 0)

    Returns:
        tf.Tensor: A single image tensor for testing
    """
    global _cached_dataset, _test_images

    # Return cached image if we have it for this seed
    if seed in _test_images:
        return _test_images[seed]

    # Load dataset only once
    if _cached_dataset is None:
        print("Loading Stanford Dogs dataset (one-time operation)...")
        _cached_dataset = tfds.load(
            'stanford_dogs', split='train',
            as_supervised=True)

    # Set seed and extract image for this specific seed
    tf.random.set_seed(seed)
    for image, _ in _cached_dataset.shuffle(10).take(1):
        _test_images[seed] = image
        break

    return _test_images[seed]


def get_dataset():
    """
    Get the full Stanford Dogs dataset.
    Caches the dataset to avoid multiple loads.

    Returns:
        tf.data.Dataset: The Stanford Dogs dataset
    """
    global _cached_dataset

    if _cached_dataset is None:
        print("Loading Stanford Dogs dataset (one-time operation)...")
        _cached_dataset = tfds.load(
            'stanford_dogs', split='train',
            as_supervised=True)

    return _cached_dataset
