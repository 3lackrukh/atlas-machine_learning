#!/usr/bin/env python3
""" Module defines the Normal Class """
e = 2.7182818285
pi = 3.1415926536


class Normal:
    """ Class defines a Normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Constructor method """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / (len(data))
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """ Calculates the z-score of a given x-value """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score """
        return self.mean + (self.stddev * z)

    def pdf(self, x):
        """ Calculates the value of the Probability Density Function at x """
        return (e ** (-((x - self.mean) ** 2) / (2 * self.stddev ** 2))) / (
            self.stddev * ((2 * pi) ** 0.5)
        )

    def cdf(self, x):
        """ Calculates the value of the CDF for a given x-value """
        # calculate taylor polynomial erf approximation
        t = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = 2 / (pi ** 0.5) * (t - (t ** 3) / 3 + (t ** 5) / 10 -
                                 (t ** 7) / 42 + (t ** 9) / 216)
        return 0.5 * (1 + erf)
