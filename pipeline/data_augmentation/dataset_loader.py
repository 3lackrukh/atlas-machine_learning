#!/usr/bin/env python3
"""Shared dataset loader for memory-efficient testing"""

import tensorflow as tf
import tensorflow_datasets as tfds

# Global variable to cache the dataset
_cached_dataset = None
_test_image = None


def get_test_image():
    """
    Get a single test image from Stanford Dogs dataset.
    Caches the dataset to avoid multiple loads.

    Returns:
        tf.Tensor: A single image tensor for testing
    """
    global _cached_dataset, _test_image

    if _test_image is not None:
        return _test_image

    if _cached_dataset is None:
        print("Loading Stanford Dogs dataset (one-time operation)...")
        tf.random.set_seed(0)
        _cached_dataset = tfds.load(
            'stanford_dogs', split='train',
            as_supervised=True)

    # Extract one image for testing
    for image, _ in _cached_dataset.shuffle(10).take(1):
        _test_image = image
        break

    return _test_image


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
        tf.random.set_seed(0)
        _cached_dataset = tfds.load(
            'stanford_dogs', split='train',
            as_supervised=True)

    return _cached_dataset
