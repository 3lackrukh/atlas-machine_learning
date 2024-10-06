#!/usr/bin/env python3
""" Module defines learning_rate_decay method """
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy.

    Parameters:
        alpha: floating point original learning rate
        decay_rate: floating point weight to determine the rate of alpha decay
        global_step: integer of elapsed gradient descent passes
        decay_step: integer of gradient descent passes between decay

    Returns:
        new_alpha: floating point updated learning rate
    """
    new_alpha = alpha / (1 + decay_rate * np.floor(global_step / decay_step))
    return new_alpha
