"""Some Machine Learning Methods done during CS-433."""
# Useful starting lines
import pickle
import datetime
import numpy as np

"""Functions"""

def check_input(y, tx, lambda_ = 0, initial_w = np.array([0,0]), max_iters = 0, gamma = 0):
    """check that all types are correct takes more time"""
    y_check = y.astype(float)
    tx_check = tx.astype(float)
    lambda__check = float(lambda_)
    w_check = initial_w.astype(float)
    max_iters_check = int(max_iters)
    gamma_check = float(gamma)
    return y_check, tx_check, lambda__check, w_check, max_iters_check, gamma_check

def compute_loss_MSE(y, tx, w):
    """calculate loss using mean squared error"""
    e = y - tx @ w
    loss = 1/(2*np.shape(tx)[0]) * e.T @ e
    return loss

def compute_loss_logistic_regression(y, tx, w):
    """calculate loss for logistic regression"""
    sigmoid = 1 / (1 + np.exp(-(tx @ w)))
    loss = -1 / np.shape(tx)[0] * np.sum((1 - y) * np.log(1 - sigmoid) + y * np.log(sigmoid))
    return loss

def shuffle_dataset(y, tx):
    """shuffling dataset"""
    # np.random.seed(1) #if commented selects every time you run a different seed
    random_shuffle = np.random.permutation(np.arange(np.shape(tx)[0]))
    shuffled_y = y[random_shuffle]
    shuffled_tx = tx[random_shuffle]
    return shuffled_y, shuffled_tx

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Computes least squares using Gradient Descent"""
    y, tx, lambda_, w, max_iters, gamma = check_input(y, tx, initial_w = initial_w,
                                                              max_iters = max_iters, gamma = gamma)

    for n_iter in range(max_iters):
        e = y - tx @ w
        gradient = -1 / np.shape(tx)[0] * tx.T @ e
        w = w - gamma * gradient

    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Computes least squares using Stochastic Gradient Descent"""
    y, tx, lambda_, w, max_iters, gamma = check_input(y, tx, initial_w=initial_w,
                                                      max_iters=max_iters, gamma=gamma)
    shuffled_y, shuffled_tx = shuffle_dataset(y, tx)

    for n_iter in range(max_iters):

        for training_example in range(np.shape(tx)[0]):
            e = shuffled_y[training_example] -shuffled_tx[training_example] @ w
            gradient = -e * shuffled_tx[training_example]
            w = w - gamma * gradient

    loss = compute_loss_MSE(shuffled_y, shuffled_tx, w)
    return w, loss

def least_squares(y, tx):
    """Computes least squares using Normal equations"""
    y, tx, lambda_, w, max_iters, gamma = check_input(y, tx)
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    loss = compute_loss_MSE(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_ ):
    """Computes ridge regression using normal equations"""
    y, tx, lambda_, w, max_iters, gamma = check_input(y, tx, lambda_=lambda_)
    w = np.linalg.inv(tx.T @ tx + lambda_ * 2 * np.shape(y)[0] * np.eye(np.shape(tx)[1])) @ tx.T @ y
    loss = compute_loss_MSE(y, tx, w) + lambda_ * w.T @ w
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Computes logistic regression using gradient descent"""
    y, tx, lambda_, w, max_iters, gamma = check_input(y, tx, initial_w=initial_w,
                                                      max_iters=max_iters, gamma=gamma)

    for n_iter in range(max_iters):
        sigmoid = 1/ (1 + np.exp(-(tx @ w)))
        gradient = -1/np.shape(tx)[0] * tx.T @ (y-sigmoid)
        w = w-gamma * gradient

    loss = compute_loss_logistic_regression(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Computes regularized logistic regression using gradient descent"""
    y, tx, lambda_, w, max_iters, gamma = check_input(y, tx, lambda_ = lambda_,initial_w=initial_w,
                                                      max_iters=max_iters, gamma=gamma)

    for n_iter in range(max_iters):
        sigmoid = 1 / (1 + np.exp(-(tx @ w)))
        gradient = -1 / np.shape(tx)[0] * tx.T @ (y - sigmoid) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = compute_loss_logistic_regression(y, tx, w) + lambda_ * w.T @ w
    return w, loss

    """Testing"""

if __name__ == "__main__":

    loop = True

    while loop:
        input_user = input('Test:\n 1 Least square \n 2 Logistic regression \n 3 end \n ')

        if int(input_user) == 1:
            """load data for least squares"""
            with open(r"test_ML_methods/data.pickle", "rb") as input_file:
                data = pickle.load(input_file)
            y = data[0]
            tx = data[1]
            test = input('ML Methods\n 1 least_squares_GD \n 2 least_squares_SGD \n 3 least_squares(Normal Equation) \n'
                         ' 4 ridge_regression(Normal Equation) \n ')

            if int(test) == 1:
                """run least_squares_GD"""
                start_time = datetime.datetime.now()
                function = least_squares_GD(y, tx, np.array([0, 0]), 14, 0.7)
                print('weights = ' , function[0], 'loss = ', function[1])
                end_time = datetime.datetime.now()
                exection_time = (end_time - start_time).total_seconds()
                print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

            elif int(test) == 2:
                """run least_squares_SGD"""
                start_time = datetime.datetime.now()
                function = least_squares_SGD(y, tx, np.array([0, 0]), 1, 0.01)
                print('weights = ', function[0], 'loss = ', function[1])
                end_time = datetime.datetime.now()
                exection_time = (end_time - start_time).total_seconds()
                print("Stochastic Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

            elif int(test) == 3:
                """run least_squares(Normal Equation)"""
                start_time = datetime.datetime.now()
                function = least_squares(y, tx)
                print('weights = ', function[0], 'loss = ', function[1])
                end_time = datetime.datetime.now()
                exection_time = (end_time - start_time).total_seconds()
                print("Normal equation least squares: execution time={t:.3f} seconds".format(t=exection_time))

            elif int(test) == 4:
                """run ridge_regression(Normal Equation)"""
                start_time = datetime.datetime.now()
                function = ridge_regression(y, tx, 0.0001)
                print('weights = ', function[0], 'loss = ', function[1])
                end_time = datetime.datetime.now()
                exection_time = (end_time - start_time).total_seconds()
                print("Regularized Normal equation least squares: execution time={t:.3f} seconds".format(t=exection_time))

            else:
                loop = False


        elif int(input_user) == 2:
            """load data for logistic regression"""
            with open(r"test_ML_methods/tx_regression.pickle", "rb") as input_file:
                tx = pickle.load(input_file)
            with open(r"test_ML_methods/y_regression.pickle", "rb") as input_file:
                y = pickle.load(input_file)
            test = input('ML Methods\n 1 logistic_regression(Gradient Descent) \n 2 reg_logistic_regression(Gradient Descent) \n ')

            if int(test) == 1:
                """run logistic_regression"""
                start_time = datetime.datetime.now()
                function = logistic_regression(y, tx, np.zeros((tx.shape[1], 1)), 1846, 1)
                print('weights = ', function[0], 'loss = ', function[1])
                end_time = datetime.datetime.now()
                exection_time = (end_time - start_time).total_seconds()
                print("logistic regression: execution time={t:.3f} seconds".format(t=exection_time))

            elif int(test) == 2:
                """run ridge_regression(Normal Equation)"""
                start_time = datetime.datetime.now()
                function = reg_logistic_regression(y, tx, 0, np.zeros((tx.shape[1], 1)), 1846, 1)
                print('weights = ', function[0], 'loss = ', function[1])
                end_time = datetime.datetime.now()
                exection_time = (end_time - start_time).total_seconds()
                print("regularized logistic regression: execution time={t:.3f} seconds".format(t=exection_time))

        elif int(input_user) ==3:
            loop = False

        else:
            loop = False