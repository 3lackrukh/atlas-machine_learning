#!/usr/bin/env python3
""" Module defines the convolve_grayscale_padding method """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
        padding: tuple of (ph, pw)
            ph: integer padding height
            pw: integer padding width

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Calculate the output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w))

    # pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph),
                     (pw, pw)),
                    mode='constant')

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # get current window
            window = padded[:, i:i+kh, j:j+kw]
            # apply kernel
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
