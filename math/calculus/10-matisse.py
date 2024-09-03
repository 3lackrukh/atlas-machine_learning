#!/usr/bin/env python3
""" Module defines the poly_derivative method """


def poly_derivative(poly):
    """ Calculates the derivative of a polynomial """
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return None
    derivative = []
    for i, coeff in enumerate(poly):
        if i == 0:
            continue
        derivative.append(i * coeff)
    if not derivative or all(coeff == 0 for coeff in derivative):
        return [0]
    return derivative
