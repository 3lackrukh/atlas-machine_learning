#!/usr/bin/env python3
""" Module defines the Binomial Class """
pi = 3.141592653589793
e = 2.718281828459045


class Binomial:
    """ Class defines a Binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ Constructor method """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / (len(data))
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def pmf(self, k):
        """
            Calculates the value of the PMF for
            a given number of 'successes'
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        coefficient = 1
        for i in range(k):
            coefficient *= (self.n - i) / (i + 1)
        return coefficient * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
            Calculates the CDF for
            a given number of 'successes'
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
