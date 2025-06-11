#!/usr/bin/env python3
"""Module defines the 3-contrast method"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: 3D tf.Tensor input image to adjust the contrast
        lower: float lower bound of the random contrast factor range
        upper: float upper bound of the random contrast factor range

    Returns:
        The contrast-adjusted image as a tf.Tensor
    """
    return tf.image.random_contrast(image, lower, upper)
