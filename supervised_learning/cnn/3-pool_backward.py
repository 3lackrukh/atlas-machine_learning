#!/usr/bin/env python3
""" Module defines the pool_backward method """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer in a neural network

    Parameters:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
        containing partial derivatives with respect to output of pooling layer
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
        kernel_shape: tuple of (kh, kw)
        containing the size of the kernel for pooling
            kh: integer kernel height
            kw: integer kernel width
        stride: tuple of (sh, sw)
        containing the strides for pooling layer
        Defaults to (1, 1)
            sh: integer stride height
            sw: integer stride width
        mode: string containing either 'max' or 'avg'
        indicating whether to perform maximum or average pooling

    Returns:
        numpy.ndarray: Partial derivatives of cost
        with respect to previous layer
    """
    # Get input dimensions
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize output array
    dA_prev = np.zeros_like(A_prev)

    # Iterate over each example, position, and channel
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):

                    # Get current window
                    window = A_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c]

                    if mode == 'max':
                        # Find max value in window
                        max_val = np.max(window)
                        # Update the corresponding elements
                        dA_prev[i, h*sh:h*sh+kh,
                                w*sw:w*sw+kw, c] += np.where(window == max_val,
                                                             dA[i, h, w, c], 0)
                    elif mode == 'avg':
                        # Calculate the average value in window
                        avg_val = dA[i, h, w, c] / (kh * kw)
                        # Update the corresponding elements
                        dA_prev[i, h*sh:h*sh+kh, w*sw:w*sw+kw, c] += avg_val

    # Return calculated gradients for previous layer
    return dA_prev
