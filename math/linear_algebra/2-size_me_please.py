#!/usr/bin/env python3
""" Module for the matrix_shape function """


def matrix_shape(matrix):
    """ Returns the shape of a given matrix as a list of integers"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if matrix:
            matrix = matrix[0]
        else:
            break
    return shape
