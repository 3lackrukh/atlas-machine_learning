#!/usr/bin/env python3
""" Module defines the conv_backward method """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding='same', stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    Parameters:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing the partial derivatives with respect to the unactivated
        output of the convolutional layer
            m: integer of examples
            h_new: integer height of the output
            w_new: integer width of the output
            c_new: integer channels in the output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            m: integer of examples
            h_prev: integer height of the previous layer
            w_prev: integer width of the previous layer
            c_prev: integer channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution
            kh: integer filter height
            kw: integer filter width
            c_prev: integer channels in the previous layer
            c_new: integer channels in the output
        b: numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution
        padding: string 'same' or 'valid', indicating the type of padding used
        stride: tuple of (sh, sw)
        containing the strides for the convolution
            sh: integer stride height
            sw: integer stride width

    Returns:
        dA_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the partial derivatives with respect to the previous layer
        dW: numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution
        db: numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution
    """
    # Get input dimensions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev + 1) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev + 1) // 2
    else:
        ph, pw = 0, 0

    # Pad input activations to match forward pass padding
    A_prev_padded = np.pad(A_prev,
                           ((0, 0), (ph, ph),
                            (pw, pw), (0, 0)),
                           mode='constant')

    # Initialize gradient matrices
    dA_prev = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Iterate over each example, position, and channel
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):

                    # Update gradient for previous layer by chain rule
                    dA_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw,
                            :] += W[:, :, :, c] * dZ[i, h, w, c]

                    # Update gradient for kernel weights by chain rule
                    dW[:, :, :, c] += A_prev_padded[i,
                                                    h*sh:h*sh+kh, w*sw:w*sw+kw,
                                                    :] * dZ[i, h, w, c]

    # Remove padding from gradient if needed
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev[:, :h_prev, :w_prev, :]

    # Return gradients for previous layer, weights, and biases
    return dA_prev, dW, db
