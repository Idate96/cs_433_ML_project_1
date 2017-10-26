"""Some Machine Learning Methods done during CS-433."""
# Useful starting lines
import pickle
import datetime
import numpy as np

"""Functions."""

def check_input(y, tx, lambda_ = 0, initial_w = 0, max_iters = 0, gamma = 0):
    """check that all types are correct takes 4 times more"""
    w = initial_w.astype(float)
    y = y.astype(float)
    tx = tx.astype(float)
    max_iters = int(max_iters)
    gamma = float(gamma)
    lambda_ = float(lambda_)
    return y, tx, lambda_, initial_w, max_iters, gamma

def compute_loss_MSE(y, tx, w):
    """calculate loss using mean squared error"""
    e = y - tx @ w
    loss = 1/(2*np.shape(tx)[0]) * e.T @ e
    return loss
def check_convergence(loss0, loss1):
    """check for convergence"""
    threshold = 1e-8
    loss = 0
    loss1 = -1 / np.shape(tx)[0] * np.sum((1 - y) * np.log(1 - sigmoid) + y * np.log(sigmoid))
    if np.abs(loss - loss1) < threshold:
        loss = loss1

    loss = loss1
    return loss
def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    y, tx, lambda_, w, max_iters, gamma = check_input(y, tx, lambda_=0, initial_w = initial_w,
                                                              max_iters = max_iters, gamma = gamma)

    for n_iter in range(max_iters):
        e = y - tx @ w
        gradient = -1 / np.shape(tx)[0] * tx.T @ e
        w = w - gamma * gradient

    loss = compute_loss_MSE(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    #shuffling dataset
    #np.random.seed(1) #if commented selects every time you run a different seed
    random_shuffle =  np.random.permutation(np.arange(np.shape(tx)[0]))
    shuffled_y = y[random_shuffle]
    shuffled_tx = tx[random_shuffle]
    
    for n_iter in range(max_iters):
        for training_example in range(np.shape(tx)[0]):
            e = shuffled_y[training_example] -shuffled_tx[training_example] @ w
            gradient = -e * shuffled_tx[training_example]
            w = w - gamma * gradient

    e = shuffled_y - shuffled_tx @ w
    loss = 1 / (2 * np.shape(tx)[0]) * e.T @ e
    return w, loss


def least_squares(y, tx):
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    e = y - tx @ w
    loss = 1 / (2 * np.shape(tx)[0]) * e.T @ e
    return w, loss


def ridge_regression(y, tx, lambda_ ):
    w = np.linalg.inv(tx.T @ tx + lambda_ * 2 * np.shape(y)[0] * np.eye(np.shape(tx)[1])) @ tx.T @ y
    e = y - tx @ w
    loss = 1 / (2 * np.shape(tx)[0]) * e.T @ e + lambda_ * w.T @ w
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    threshold = 1e-8
    loss = 0

    for n_iter in range(max_iters):
        sigmoid = 1/ (1 + np.exp(-(tx @ w)))
        gradient = -1/np.shape(tx)[0] * tx.T @ (y-sigmoid)
        w = w-gamma * gradient
        loss1 = -1 / np.shape(tx)[0] * np.sum((1 - y) * np.log(1 - sigmoid) + y * np.log(sigmoid))
        if np.abs(loss - loss1) < threshold:
            loss = loss1
            break
        loss = loss1
    return w, loss

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        sigmoid = 1 / (1 + np.exp(-(tx @ w)))
        gradient = -1 / np.shape(tx)[0] * tx.T @ (y - sigmoid) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = -1 / np.shape(tx)[0] * np.sum((1 - y) * np.log(1 - sigmoid) + y * np.log(sigmoid)) + lambda_ * w.T @ w

    return w, loss


 #test data for least square
with open(r"data.pickle", "rb") as input_file:
    data = pickle.load(input_file)
y = data[0]
tx = data[1]
print(y)
print(tx[0])
'''
#test data for logistic regression
with open(r"tx_regression.pickle", "rb") as input_file:
    tx = pickle.load(input_file)
with open(r"y_regression.pickle", "rb") as input_file:
    y = pickle.load(input_file)
'''

#check least_squares_GD()
start_time = datetime.datetime.now()
print(least_squares_GD(y,tx,np.array([0,0]), 50, 0.7))
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

'''check least_squares_SGD()
start_time = datetime.datetime.now()
print(least_squares_SGD(y,tx,np.array([0,0]), 1, 0.01))
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("Stochastic Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))
'''

''' least squares
start_time = datetime.datetime.now()
print(least_squares(y,tx))
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("Normal equation least squares: execution time={t:.3f} seconds".format(t=exection_time))
'''
''' ridge_regression
start_time = datetime.datetime.now()
print(ridge_regression(y,tx, 10000))
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("Regularized Normal equation least squares: execution time={t:.3f} seconds".format(t=exection_time))
'''
'''logistic regression
start_time = datetime.datetime.now()
print(logistic_regression(y,tx, np.zeros((tx.shape[1], 1)), 10000, 1))
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("logistic regression: execution time={t:.3f} seconds".format(t=exection_time))
'''

'''reg_logistic regression
start_time = datetime.datetime.now()
print(reg_logistic_regression(y,tx, 0.05, np.zeros((tx.shape[1], 1)), 401, 0.1))
end_time = datetime.datetime.now()
exection_time = (end_time - start_time).total_seconds()
print("regularized logistic regression: execution time={t:.3f} seconds".format(t=exection_time))
'''


