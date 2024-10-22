#!/usr/bin/env python3
""" Module defines the convolve_grayscale_valid method """
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

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
    output = np.zeros((m, h, w))

    # calculate padding
    pad_h = kh // 2
    pad_w = kw // 2

    # pad images
    padded = np.pad(images,
                    ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant')

    # Perform convolution
    for i in range(h):
        for j in range(w):
            # get current window
            window = padded[:, i:i+kh, j:j+kw]
            # apply kernel
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
