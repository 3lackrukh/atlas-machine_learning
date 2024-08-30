#!/usr/bin/env python3
""" Module defines the np_cat method"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ numpy concatenate Concatenates two matrices along specified axis """
    return np.concatenate((mat1, mat2), axis=axis)
