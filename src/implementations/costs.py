# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np
def compute_loss(y, tx, w, type='mse'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = tx.shape[0]
    if type == 'mse':
        cost = 1 / N * (y - tx @ w).T @ (y - tx @ w)
    elif type == 'mae':
        cost = 1/N * np.sum(np.abs((y - tx @ w)))
    else:
        raise ValueError("type not recognised")
    return cost