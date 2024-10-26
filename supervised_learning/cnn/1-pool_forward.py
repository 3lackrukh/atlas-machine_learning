#!/usr/bin/env python3
""" Module defines the pool_forward method """
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    Parameters:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            m: integer of images
            h_prev: integer height in pixels
            w_prev: integer width in pixels
            c_prev: integer color channels in images
        kernel_shape: tuple containing the kernel size for pooling
            kh: integer height of the kernel
            kw: integer width of the kernel
        stride: tuple of (sh, sw)
            sh: integer stride height
            sw: integer stride width
        mode: string 'max' or 'avg'
            'max': max pooling
            'avg': average pooling

    Returns:
        A numpy.ndarray containing the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Update dimensions for output array
    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w, c_prev))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            # get current window
            window = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            # apply kernel
            if mode == 'max':
                output[:, i, j, :] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(window, axis=(1, 2))

    return output
