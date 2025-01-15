#!/usr/bin/env python3
""" Module defines the determinant, submatrix, and inverse methods """


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


def inverse(matrix):
    """
    Calculates the adjugate matrix of a matrix.

    Parameters:
    - matrix: list of lists whose cofactor matrix should be calculated

    Returns:
    - list of lists: cofactor matrix of the matrix
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

    cofactor_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sub = submatrix(matrix, i, j)
            cofactor_matrix[i][j] = (-1) ** (i + j) * determinant(sub)

    adjugate_matrix = [[cofactor_matrix[j][i]
                        for j in range(n)]
                       for i in range(n)]

    det = determinant(matrix)
    if det == 0:
        return None

    inverse_matrix = [[adjugate_matrix[i][j] / det
                       for j in range(n)]
                      for i in range(n)]

    return inverse_matrix
