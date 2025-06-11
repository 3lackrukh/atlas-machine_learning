#!/usr/bin/env python3
"""Module defines the 5-hue method"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    Args:
        image: 3D tf.Tensor input image to adjust the hue
        delta: float amount to change the hue

    Returns:
        The hue-adjusted image as a tf.Tensor
    """
    return tf.image.adjust_hue(image, delta)
