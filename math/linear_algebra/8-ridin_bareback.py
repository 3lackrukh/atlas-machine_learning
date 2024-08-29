#!/usr/bin/env python3
""" Module defines the mat_mul fucntion """
matrix_shape = __import__('2-size_me_please').matrix_shape


def mat_mul(mat1, mat2):
    """ Multiplies two 2D matrices """
    if matrix_shape(mat1)[1] != matrix_shape(mat2)[0]:
        return None
    new_mat = []
    for row in mat1:
        new_row = []
        for col in zip(*mat2):
            new_row.append(sum(x * y for x, y in zip(row, col)))
        new_mat.append(new_row)
    return new_mat
