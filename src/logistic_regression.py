import numpy as np
from src.utils import batch_iter, dataloader, split_data, standardize, xavier_init, adam
import matplotlib.pyplot as plt
num_epochs = 300
batch_size = 300
learning_rate = 10**-4

x, y = dataloader(mode='train', reduced=False)
x = standardize(x)
train_dataset, test_dataset = split_data(x, y, ratio=0.9)
test_data, test_target = test_dataset
train_data, train_target = train_dataset
print(np.shape(train_data))
num_batches = int(np.shape(train_data)[0]/batch_size)
# till now up to degree two its fine + x**3 (no mixed cubic terms
def build_polynomial(x):
    base_mixed = np.zeros((np.shape(x)[0],int(np.shape(x)[1]*(np.shape(x)[1]-1)/2)))
    # base_mixed_cube = np.zeros((np.shape(x)[0], int(np.shape(x)[1]**2)))
    bias = np.ones(np.shape(x)[0])
    counter = 0
    # gaussian_base = np.zeros((np.shape(x)[0],int(np.shape(x)[1]*(np.shape(x)[1]-1)/2)))
    for i in range(np.shape(x)[1]):
        for j in range(i):
            base_mixed[:, counter] = x[:, i] * x[:, j]
            # gaussian_base[:, counter] = np.exp(-(x[:, i] - x[:, j])**2/(2*0.25))
            counter += 1

    counter = 0
    base_mixed_cube = np.zeros((np.shape(x)[0], np.shape(x)[1]-2))

    # for i in range(np.shape(x)[1]):
    #     for j in range(np.shape(x)[1]):
    #         base_mixed_cube[:, counter] = x[:, i]**2 * x[:, j]
    #
    base = np.hstack((bias[:, np.newaxis], x, base_mixed, x**2, x**3))
    return base

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def loss_mse(weights, x, target, lambda_=0):
    g_x = sigmoid(x @ weights)
    return 1/np.shape(target)[0] * (target - g_x).T @ (target - g_x)

def gradient_mse(weights, x, target, lambda_= 0):
    g_x = sigmoid(x @ weights)
    return x.T @ ((g_x - target) * g_x * (1 - g_x))

def loss_mse_reg(weights, x, target, lambda_ = 0):
    g_x = sigmoid(x @ weights)
    return 1/np.shape(target)[0] * (target - g_x).T @ (target - g_x) + lambda_ * weights.T @ weights

def gradient_mse_reg(weights, x, target, lambda_ = 0):
    g_x = sigmoid(x @ weights)
    return x.T @ ((g_x - target) * g_x * (1 - g_x)) + lambda_ * weights

def loss_ce(weights, x, target, lambda_=0):
    g_x = np.clip(sigmoid(x @ weights), 0.0001, 0.9999)
    loss = -1/np.shape(x)[0] * np.sum(target * np.log(g_x) + (1 - target)*np.log(1 - g_x))  + \
         lambda_ * weights.T @ weights
    return loss

def gradient_ce(weights, x, target, lambda_):
    g_x = sigmoid(x @ weights)
    return -1/np.shape(target)[0] * x.T @ (target - g_x) + lambda_ * weights

def train_logistic_regression(loss_func, grad_func, lambda_ = 10**-3, show_every=10):
    global learning_rate
    iter_num = 1
    m, v = 0, 0
    # m, v = 0, 0
    poly_train_data = build_polynomial(train_data)
    weights = xavier_init(np.shape(poly_train_data[1]))
    # weights = np.zeros(np.shape(poly_train_data)[1])
    for epoch in range(num_epochs):
        if epoch % 90 == 0:
            learning_rate *= 0.5
        for batch_label, batch_input in batch_iter(
                train_target, train_data, batch_size, num_batches=num_batches):
            batch_input = build_polynomial(batch_input)
            grad = grad_func(weights, batch_input, batch_label, lambda_)
            # weights, m, v = adam(weights, m, v, 0.9, 0.999, learning_rate, grad, iter_num)
            weights -= learning_rate*grad
            iter_num += 1
        train_loss = loss_func(weights, poly_train_data, train_target, lambda_)
        if epoch % show_every == 0 or epoch == num_epochs - 1:
            print("Epoch : ", epoch)
            print("Train loss : ", train_loss)
            # print('Weights :', weights)
            weights, accuracy, test_loss = test_logistic_regression(weights, loss_func, lambda_)
    return weights, accuracy, train_loss, test_loss

def test_logistic_regression(weights, loss_func, lambda_):
    poly_test_data = build_polynomial(test_data)
    loss = loss_func(weights, poly_test_data, test_target, lambda_)
    output = sigmoid(poly_test_data @ weights)
    predicted = output > 0.5
    correct = np.sum(predicted == test_target)
    accuracy = correct/np.shape(test_data)[0]
    print("Test loss :", loss)
    print('Test accuracy :', accuracy)
    return weights, accuracy, loss

def find_best_lambda(loss_func, grad_func):
    lambdas = np.logspace(-5, 0, 20)
    weights_history = []
    accuracies = []
    train_losses = []
    test_losses = []
    best_weigths = None
    best_accurary = 0
    best_combination = 0
    for idx, lambda_ in enumerate(lambdas):
        weights, accuracy, train_loss, test_loss = train_logistic_regression(
                                                loss_func, grad_func, lambda_)
        weights_history.append(weights)
        accuracies.append(accuracy)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if accuracy > best_accurary:
            best_accurary = accuracy
            best_weigths = weights
            best_combination = idx

    print('best combination lambda : ', lambdas[best_combination])
    return lambdas, best_weigths, best_accurary, test_losses, train_losses, best_combination

def plot(x, train_loss, test_loss):
    plt.plot(x, train_loss, label='train loss')
    plt.plot(x, test_loss, label='test loss')
    plt.show()




if __name__ == '__main__':
    train_logistic_regression(loss_ce, gradient_ce, lambda_= 0.01)
    # lambdas, best_weigths, best_accurary, test_losses, train_losses, \
    # best_combination = find_best_lambda(loss_mse_reg, gradient_mse_reg)
    # plot(lambdas, train_losses, test_losses)

    best_lambda = 0.0012742749857




