# -*- coding: utf-8 -*-
"""Gradient Descent"""
import datetime
import numpy as np

def compute_gradient(y, tx, w, type='mse'):
    """Compute the gradient."""
    if type == 'mse':
        gradient = -2/np.shape(tx)[0] * tx.T @ (y - tx @ w)
    elif type == 'mae':
        gradient = -1 / np.shape(tx)[0] * tx.T @ np.sign(y - tx @ w)
    else:
        raise ValueError('type not implemented')
    assert np.shape(w) == gradient
    return gradient


def gradient_descent(y, tx, initial_w, loss_func, gradient_func, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = loss_func(y, tx, w)
        dw = gradient_func(y, tx, w)
        w = w - gamma * dw
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def train():
    max_iters = 50
    gamma = 0.1

    # Initialization
    w_initial = np.array([0, 0])

    # Start gradient descent.
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))