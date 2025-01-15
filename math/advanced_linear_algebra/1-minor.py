#!/usr/bin/env python3
""" Module defines the minor method """


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

    # Copy matrix
    minor_matrix = [_[:] for _ in matrix]

    for i in range(n):
        for j in range(n):
            # Get submatrix removing row i and column j
            submatrix = [
                row[:j] + row[j+1:]
                for row in (matrix[:i] + matrix[i+1:])
            ]

            # Calculate determinant for 2x2 matrix
            if len(submatrix) == 1:
                minor_matrix[i][j] = submatrix[0][0]
            elif len(submatrix) == 2:
                minor_matrix[i][j] = (submatrix[0][0] * submatrix[1][1] -
                                      submatrix[0][1] * submatrix[1][0])
            else:
                # Calculate determinant for larger matrix
                det = 0
                for x in range(len(submatrix)):
                    sub = [row[:x] + row[x+1:] for row in submatrix[1:]]
                    det += ((-1) ** x) * submatrix[0][x] * minor(sub)[0][0]
                minor_matrix[i][j] = det

    return minor_matrix
