#!/usr/bin/env python3
""" Module defines the pool method """
import numpy as np


def pool(images, kernel, stride=(1, 1), mode='max'):
    """
    Performs a same convolution on grayscale images.

    Parameters:
        images: numpy.ndarray of shape (m, h, w, c) containing images.
            m: integer of images
            h: integer height in pixels
            w: integer width in pixels
            c: integer color channels in images
        kernel: tuple containing the kernel shape for the
        convolution.
            kh: integer height of the kernel
            kw: integer width of the kernel
        stride: tuple of (sh, sw)
            sh: integer stride height
            sw: integer stride width
        mode: string 'max' or 'avg'
            'max': max pooling
            'avg': average pooling

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel
    sh, sw = stride

    # Update dimensions for output array
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w, c))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # get current window
            window = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            # apply kernel
            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output
