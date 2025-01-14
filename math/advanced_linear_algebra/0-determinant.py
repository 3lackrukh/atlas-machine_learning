#!/usr/bin/env python3
""" Module defines the determinant method """


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Parameters:
    - matrix: list of lists whose determinant should be calculated

    Returns:
    - int/float: determinant of the matrix
    """
    if matrix == [[]]:
        return 1

    if not matrix or not isinstance(matrix, list)\
            or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for x in range(n):
        submatrix = [row[:x] + row[x+1:] for row in matrix[1:]]
        det += (-1) ** x * matrix[0][x] * determinant(submatrix)

    return det
