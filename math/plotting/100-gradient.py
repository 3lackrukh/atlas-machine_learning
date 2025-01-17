#!/usr/bin/env python3
""" Module defines the gradient method """
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """ Creates a scatter plot of sampled elevations on a mountain """

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, c=z, cmap="viridis")
    plt.colorbar(label="elevation (m)")
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.title("Mountain Elevation")
    plt.show()


if __name__ == "__main__":
    gradient()
