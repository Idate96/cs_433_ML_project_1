import sys
import os
from src.utils import create_csv_submission
import numpy as np
import pickle
import csv
from src.utils import sigmoid, batch_iter, dataloader, standardize, \
    build_polynomial, split_data_k_fold
sys.path.append(os.getcwd())


class Config(object):
    """Configuration object for the classifiers
    batch sixe, number of epochs (the amount of time the program goes through de dataset, learning rate is the step
    of the gradient, the lambda and mode can be either cross-validation or test"""

    def __init__(self, batch_size, num_epochs, learning_rate, lambda_, mode='cv'):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.mode = mode


class LogisticClassifier(object):
    def __init__(self, config, train_set=(None, None), test_set=(None, None), label='0'):
        self.config = config
        # construct non linear features
        self.train_data, self.train_labels = train_set
        if self.train_data is not None:
            self.weights = np.zeros((np.shape(self.train_data)[1]))
        self.best_weights = None

        self.test_data, self.test_labels = test_set
        # self.weights = xavier_init(np.shape(self.train_data))
        self.train_losses = []
        self.train_loss = None
        self.test_losses = []
        self.test_loss = None
        self.accuracy = 0
        self.best_accuracy = 0
        self.train_accuracy = 0
        self.test_predictions = None
        self.train_predictions = None
        self.label = label

    def reset(self):
        self.train_losses = []
        self.train_loss = None
        self.test_losses = []
        self.test_loss = []
        self.accuracy = 0
        self.train_accuracy = 0
        self.weights = np.zeros((np.shape(self.train_data)[1]))
        self.test_predictions = None

    def __call__(self, input):
        return sigmoid(np.dot(input, self.weights))

    def loss(self, output, target, sample_weights=1):
        loss = 1 / np.shape(target)[0] * np.dot((target - output).T, (
        sample_weights * (target - output))) + self.config.lambda_ * np.dot(self.weights.T,
                                                                            self.weights)
        return loss

    def grad(self, data_batch, target_batch, sample_weights=1):
        return np.dot(data_batch.T, (
        (self(data_batch) - target_batch) * sample_weights * self(data_batch) * (
        1 - self(data_batch)))) + self.config.lambda_ * self.weights

    def sdg(self, param, data, target, learning_rate):
        param -= learning_rate * self.grad(data, target)
        return param

    def train(self, show_every=10):
        reduction_factor = 1
        num_batches = int(np.shape(self.train_data)[0] / self.config.batch_size)
        for epoch in range(self.config.num_epochs):
            if epoch % 50 == 0:
                reduction_factor *= 0.5
            for batch_label, batch_input in batch_iter(self.train_labels, self.train_data,
                    self.config.batch_size, num_batches=num_batches):
                self.weights = self.sdg(self.weights, batch_input, batch_label,
                                        self.config.learning_rate * reduction_factor)

            self.train_loss = self.loss(self(self.train_data), self.train_labels)
            if epoch % show_every == 0 or epoch == self.config.num_epochs - 1:
                self.train_predictions = self.predict(self(self.train_data))
                correct = np.sum(self.train_predictions == self.train_labels)
                self.train_accuracy = correct / np.shape(self.train_data)[0]

                print("Epoch : ", epoch)
                print("Train loss : ", self.train_loss)
                print("Train accuracy : ", self.train_accuracy)
                if self.config.mode == 'cv':
                    self.test()

    def test(self):
        output = self(self.test_data)
        self.test_loss = self.loss(output, self.test_labels)
        self.test_losses.append(self.test_loss)
        self.test_predictions = self.predict(output)
        correct = np.sum(self.test_predictions == self.test_labels)
        self.accuracy = correct / np.shape(self.test_data)[0]
        if self.accuracy > self.best_accuracy:
            self.best_accuracy = self.accuracy
            self.best_weights = self.weights
        print("Test loss :", self.test_loss)
        print('Test accuracy :', self.accuracy)

    def predict(self, output):
        return output > 0.5

    def save(self):
        with open(r'config/weights' + self.label + '.p', "wb") as file:
            pickle.dump(self.weights, file)

    def load_weights(self):
        self.weights = pickle.load(open('config/weights' + self.label + '.p', 'rb'))

    def export_predictions(self):
        with open('prediction/submission.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for i in range(len(self.test_predictions)):
                writer.writerow([str(i) + ", " + self.test_predictions[i]])


class EnsembleClassifiers(object):
    def __init__(self, config, x, y, num_classifiers, classifier, label='0'):
        self.train_data = x
        self.train_label = y
        self.config = config
        self.classifiers = []
        for i in range(num_classifiers):
            if config.mode == 'cv':
                train_set, test_set = split_data_k_fold(x, y, i % 10, k=10)
                self.classifiers.append(classifier(config, train_set, test_set))
            else:
                # x, y = randomize_samples(x, y)
                self.classifiers.append(
                    classifier(config, train_set=(self.train_data, self.train_label)))
        self.classifier_weights = np.ones(num_classifiers)
        self.test_predictions = None
        self.label = label

    def __call__(self, data):
        print(self.classifiers[0](data))
        output = np.zeros(np.shape(data)[0])
        for classifier in self.classifiers:
            output += 1 / len(self.classifiers) * classifier(data)
        return output

    def predict(self, output):
        y_pred = np.zeros(np.shape(output)[0])
        y_pred[np.where(output <= 0.5)] = -1
        y_pred[np.where(output > 0.5)] = 1
        return y_pred

    def train(self):
        for classifier in self.classifiers:
            classifier.train()
        if self.config.mode == 'cv':
            self.test()

    def test(self):
        self.accuracy = 0
        for classifier in self.classifiers:
            self.accuracy += 1 / len(self.classifiers) * classifier.best_accuracy
        print('Test ensemble accuracy :', self.accuracy)

    def save(self):
        weights = np.zeros((len(self.classifiers), np.shape(self.train_data)[1]))
        for i, classifier in enumerate(self.classifiers):
            weights[i] = classifier.weights
        with open(r'config/' + self.label + '.p', "wb") as file:
            pickle.dump(weights, file)

    def load_weights(self):
        with open(r'config/' + self.label + '.p', "rb") as file:
            weights = pickle.load(file)
        print(np.shape(weights))
        for i, classifier in enumerate(self.classifiers):
            classifier.weights = weights[i]


def find_best_regularizer(model_class, lambdas):
    x, y = dataloader(mode='train', reduced=False)
    x = standardize(x)
    best_lambda = 0
    best_accuracy = 0
    for idx, lambda_ in enumerate(lambdas):
        print('Ensemble nr ' + str(idx) + 30 * '=')
        config = Config(batch_size=200, num_epochs=100, learning_rate=5 * 10 ** -4, lambda_=lambda_)
        ensemble = EnsembleClassifiers(config, build_polynomial(x), y, 10, LogisticClassifier,
                                       label='ensemble_' + str(idx))
        ensemble.train()
        print("ensemble accuracy " + str(ensemble.accuracy) + 30 * "=")
        if ensemble.accuracy > best_accuracy:
            best_accuracy = ensemble.accuracy
            best_lambda = lambda_
        print("best_lambda :", best_lambda)
