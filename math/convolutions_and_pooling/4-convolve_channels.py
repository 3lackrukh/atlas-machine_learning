#!/usr/bin/env python3
""" Module defines the convolve_channels method """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a same convolution on images with channels.

    Parameters:
        images: numpy.ndarray of shape (m, h, w, c) contains grayscale images.
            m: integer of images
            h: integer height in pixels
            w: integer width in pixels
            c: integer channels in the image
        kernel: numpy.ndarray of shape (kh, kw, kc) containing the kernel for the
        convolution.
            kh: integer height of the kernel
            kw: integer width of the kernel
            kc: integer channels in the image
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
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    # confirm kernel dimensions
    if kc != c:
        raise ValueError("kernel channels must match image channels")

    # Calculate padding dimensions
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + (kh % 2 == 0)
        pw = ((w - 1) * sw + kw - w) // 2 + (kw % 2 == 0)
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
                     (pw, pw), (0, 0)),
                    mode='constant')

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # get current window
            window = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            # apply kernel
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))

    return output
