#!/usr/bin/env python3
""" Module defines the determinant, submatrix, and minor methods """


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


def submatrix(matrix, i, j):
    """
    Creates a submatrix by removing the specified row and column.

    Parameters:
    - matrix: list of lists representing the input matrix
    - i: row index to remove
    - j: column index to remove

    Returns:
    - list of lists: submatrix with row and column removed
    """
    return [[matrix[r][c]
            for c in range(len(matrix)) if c != j]
            for r in range(len(matrix)) if r != i]


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.

    Parameters:
    - matrix: list of lists whose minor matrix should be calculated

    Returns:
    - list of lists: minor matrix of the matrix
    """
    if not isinstance(matrix, list)\
            or matrix == []\
            or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if not all(len(row) == n for row in matrix)\
            or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Get submatrix removing row i and column j
            sub = submatrix(matrix, i, j)
            if sub == [[]]:
                return [[1]]
            minor_matrix[i][j] = determinant(sub)

    return minor_matrix
