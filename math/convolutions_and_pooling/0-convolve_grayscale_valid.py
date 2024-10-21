#!/usr/bin/env python3
""" Module defines the convolve_grayscale_valid method """
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
        images: numpy.ndarray of shape (m, h, w) containing grayscale images.
            m: integer of images
            h: integer height in pixels
            w: integer width in pixels
        kernel: numpy.ndarray of shape (kh, kw) containing the kernel for the
        convolution.
            kh: integer height of the kernel
            kw: integer width of the kernel

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Initialize the output array
    output = np.zeros((m, h - kh + 1, w - kw + 1))

    # Perform convolution
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
