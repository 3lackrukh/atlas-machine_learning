#!/usr/bin/env python3
""" Module defines the early_stopping method """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.

    Parameters:
        cost: floating point current validation cost of the neural network.
        opt_cost: floating point lowest recorded validation cost of the neural network.
        threshold: threshold used for early stopping.
        patience: integer count used for early stopping.
        count: integer of how long the validation cost has not improved.

    Returns:
        Boolean whether the network should be stopped early,
        followed by the updated count.
    """
    if cost < opt_cost - threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count