#!/usr/bin/env python3
""" Module defines the convolve_grayscale method """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
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
        padding: either a tuple of (ph, pw), or string 'same' or 'valid'.
            ph: integer padding height
            pw: integer padding width
            'same': output has same height and width as input
            'valid': no padding
        stride: tuple of (sh, sw)
            sh: integer stride height
            sw: integer stride width

    Returns:
        A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    
    # Calculate padding dimensions
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0

    # Update dimensions for output array
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1
    
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
            window = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            # apply kernel
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
