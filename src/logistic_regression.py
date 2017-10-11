import numpy as np
from utils import batch_iter, dataloader, split_data, standardize

num_epochs = 10
batch_size = 100
learning_rate = 10**-3

x, y = dataloader(mode='train', reduced=False)
x = standardize(x)
train_dataset, test_dataset = split_data(x, y, ratio=0.9)
test_data, test_target = test_dataset
train_data, train_target = train_dataset
num_batches = int(np.shape(train_data)[0]/batch_size)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def loss_mse(weights, x, target):
    g_x = sigmoid(x @ weights)
    return 1/np.shape(target)[0] * (target - g_x).T @ (target - g_x)

def gradient_mse(weights, x, target):
    g_x = sigmoid(x @ weights)
    return (g_x - target) * g_x * (1 - g_x) * x

def loss_ce(weights, x, target):
    g_x = sigmoid(x @ weights)
    return 1/np.shape(x)[0] * np.sum(target * np.log(g_x) + (1-target)*np.log(1-g_x))

def gradient_ce(weights, x, target):
    g_x = sigmoid(x @ weights)
    return 1/np.shape(target)[0] * x.T @ (target - g_x)

def train_logistic_regression(loss_func, grad_func):
    weights = np.zeros(30)
    for epoch in range(num_epochs):
        for batch_label, batch_input in batch_iter(
                train_target, train_data, batch_size, num_batches=num_batches):
            grad = grad_func(weights, batch_input, batch_label)
            weights -= learning_rate * grad
        train_loss = loss_func(weights, train_data, train_target)
        print("Epoch : ", epoch)
        print("Train loss : ", train_loss)
        test_logistic_regression(weights, loss_func)


def test_logistic_regression(weights, loss_func):
    loss = loss_func(weights, test_data, test_target)
    output = sigmoid(test_data @ weights)
    predicted = output > 0.5
    correct = np.sum(predicted == test_target)
    accuracy = correct/np.shape(test_data)[0]
    print("Test loss :", loss)
    print('Test accuracy :', accuracy)

if __name__ == '__main__':
    train_logistic_regression(loss_ce, gradient_ce)

