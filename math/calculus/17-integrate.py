#!/usr/bin/env python3
""" Module defines the poly_integral method """


def poly_integral(poly, C=0):
    """ Calculates the integral of a polynomial """
    if not isinstance(poly, list) or not isinstance(C, (int, float)):
        return None
    if not poly:
        return None
    integral = [C]
    for i, coeff in enumerate(poly):
        if coeff % (i + 1) == 0:
            # use integer division to avoid float imprecision
            integral.append(coeff // (i + 1))
        else:
            # use regular division for fractional coefficient
            integral.append(coeff / (i + 1))
    # remove trailing zeros
    if integral[-1] == 0:
        integral.pop()
    return integral
