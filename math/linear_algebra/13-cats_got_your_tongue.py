#!/usr/bin/env python3
import numpy as np
""" Module defines the np_cat method"""


def np_cat(mat1, mat2, axis=0):
    """ numpy concatenate Concatenates two matrices along specified axis """
    return np.concatenate((mat1, mat2), axis=axis)
