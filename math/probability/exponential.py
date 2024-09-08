#!/usr/bin/env python3
""" Module defines the Exponential class """


class Exponential:
    """ Class defines an exponential function """

    def __init__(self, data=None, lambtha=1):
        """ Initializes the exponential function """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))