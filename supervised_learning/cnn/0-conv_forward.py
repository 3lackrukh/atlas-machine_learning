#!/usr/bin/env python3
""" Module defines the conv_forward method """
import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1,1)):
    """
    Performs forward propagation over a convolutional layer
    of a neural network.
    
    Parameters:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer.
            m: integer of examples
            h_prev: integer height of the previous layer
            w_prev: integer width of the previous layer
            c_prev: integer number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution.
            kh: integer filter height
            kw: integer filter width
            c_prev: integer of channels in the previous layer
            c_new: integer of channels in the output
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution.
        activation: activation function applied to the convolution.
        padding: string indicating the type of padding used.
            'same': output has same height and width as input
            'valid': no padding
        stride: tuple of (sh, sw)
            sh: integer stride height
            sw: integer stride width

    Returns:
        numpy.ndarray output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # calculate padding dimensions
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev + 1) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev + 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0

    # Pad examples
    padded = np.pad(A_prev,
                    ((0, 0), (ph, ph),
                     (pw, pw), (0, 0)),
                    mode='constant')

    # Update dimensions for output array
    output_h = (h_prev + 2 * ph - kh) // sh + 1
    output_w = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c_new):

                # Define current window
                window = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

                # Apply kernel
                output[:, i, j, k] = np.sum(window * W[..., k],
                                            axis=(1, 2, 3)) + b[..., k]

    # Apply activation function and return outpur
    return activation(output)
