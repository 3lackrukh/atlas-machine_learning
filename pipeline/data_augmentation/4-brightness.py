#!/usr/bin/env python3
"""Module defines the 4-brightness method"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: 3D tf.Tensor input image to adjust the brightness
        max_delta: float maximum brightness adjustment factor

    Returns:
        The brightness-adjusted image as a tf.Tensor
    """
    return tf.image.random_brightness(image, max_delta)
