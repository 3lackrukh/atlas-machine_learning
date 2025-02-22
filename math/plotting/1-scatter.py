#!/usr/bin/env python3
""" Module plots x ↦ y as a scatter plot"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    x-axis: Height (in)
    y-axis: Weight (lbs)
    title: Men's Height vs Weight
    data plotted as magenta points
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.title("Men's Height vs Weight")
    plt.scatter(x, y, c='magenta', marker='o')
    plt.show()


if __name__ == "__main__":
    scatter()
