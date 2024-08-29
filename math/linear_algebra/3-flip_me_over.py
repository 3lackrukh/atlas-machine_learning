#!/usr/bin/env python3
""" Module defines the matrix_transpose function """


def matrix_transpose(matrix):
    """ Returns the transpose of a 2D matrix """
    transpose = []
    for i in range(len(matrix[0])):
        row = []
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        transpose.append(row)
    return transpose
