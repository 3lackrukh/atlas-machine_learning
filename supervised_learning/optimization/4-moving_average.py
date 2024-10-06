#!/usr/bin/env python3
""" Module defines the moving_average method """


def moving_average(data, beta):
    """
    Calculates the moving average of a list of data points.

    Parameters:
        data: list of data points.
        beta: float weight for moving average.

    Returns:
        moving_average: list of moving averages.
    """
    moving_average = []
    avg = 0

    for i, value in enumerate(data):
        avg = beta * avg + (1 - beta) * value
        moving_average.append(avg / (1 - beta ** (i + 1)))

    return moving_average
