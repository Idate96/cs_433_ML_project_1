# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, grad_func, loss_func, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    num_batches = int(np.shape(tx)[0]/batch_size)
    for epoch_num in range(max_iters):
        for batch_idx in range(num_batches):
            tx_batch = tx[batch_idx * batch_size : (batch_idx + 1)*batch_size]
            y_batch = y[batch_idx * batch_size : (batch_idx + 1)*batch_size]
            dw = grad_func(y_batch, tx_batch, w)
            w = w - gamma * dw
            # store w and loss
        loss = loss_func(y, tx, w)
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=epoch_num, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws