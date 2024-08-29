#!/usr/bin/env python3
""" Module for the matrix_shape function """


def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if matrix:
            matrix = matrix[0]
        else:
            break
    return shape
