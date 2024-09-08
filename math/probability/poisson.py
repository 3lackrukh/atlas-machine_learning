#!/usr/bin/env python3
""" Module defines the Poisson class """


class Poisson:
    """ Class defines a Poisson process """
    def __init__(self, data=None, lambtha=1):
        """ Initializes the Poisson process """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data)/len(data))