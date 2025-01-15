#!/usr/bin/env
import numpy as np
"""
Module defines the definiteness method
determining the definiteness of a matrix without conditionals
"""


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix

    Parameters:
        matrix (numpy.ndarray): the matrix to calculate the definiteness of

    Returns:
        str: the definiteness of the matrix

    Raises:
        TypeError: if matrix is not a numpy.ndarray
    """
    try:
        validity = (
            matrix.ndim == 2 and
            matrix.shape[0] == matrix.shape[1] and
            np.allclose(matrix, matrix.T)
        )

        eigenvalues = np.linalg.eigvals(matrix) * validity

        definiteness_map = {
            (True, False, False, False): "Positive definite",
            (False, True, False, False): "Positive semi-definite",
            (False, False, True, False): "Negative definite",
            (False, False, False, True): "Negative semi-definite"
        }

        conditions = (
            np.all(eigenvalues > 0),
            np.all(eigenvalues >= 0) and not np.all(eigenvalues > 0),
            np.all(eigenvalues < 0),
            np.all(eigenvalues <= 0) and not np.all(eigenvalues < 0)
        )

        result_map = {
            True: "Indefinite",
            False: None
        }
        return definiteness_map.get(conditions, result_map[validity])

    except AttributeError:
        raise TypeError("matrix must be a numpy.ndarray")
    except Exception:
        return None
