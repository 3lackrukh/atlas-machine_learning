#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
""" demonstrates the creation of a line plot using Matplotlib. """


def line():
    x = np.arange(0, 11)
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)
    plt.show()
