#!/usr/bin/env python3
""" Module defines the add_matrices2D function """


matrix_shape = __import__('2-size_me_please').matrix_shape

def add_matrices2D(mat1, mat2):
    """ Function adds two matrices element-wise """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[i])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
