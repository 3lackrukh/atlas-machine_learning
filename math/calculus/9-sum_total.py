#!/usr/bin/env python3
""" Module defines the summation_i_squared method """

def summation_i_squared(n):
    """ Calculates the sum of squares of numbers from 1 to n """
    total = 0
    for i in range(1, n + 1):
        total += i ** 2
    return total
