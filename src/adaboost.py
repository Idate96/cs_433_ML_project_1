import sys
import os
sys.path.append(os.getcwd())
from src.utils import sigmoid, batch_iter, dataloader, split_data,\
    standardize, xavier_init, build_polynomial, split_data_k_fold, randomize_samples
from src.utils import create_csv_submission
import numpy as np
import pickle
import csv

class Config(object):
    """Configuration object for the classifiers"""
    def __init__(self, batch_size, num_epochs, learning_rate, lambda_, mode = 'cv'):
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
        loss = 1 / np.shape(target)[0] * np.dot((target - output).T,
                (sample_weights * (target - output))) + self.config.lambda_ *\
                                                        np.dot(self.weights.T, self.weights)
        return loss

    def grad(self, data_batch, target_batch, sample_weights=1):
        return np.dot(data_batch.T, ((self(data_batch) - target_batch) * sample_weights *
                self(data_batch) * (1 - self(data_batch)))) + \
               self.config.lambda_ * self.weights

    def sdg(self, param, data, target, learning_rate):
        param -= learning_rate * self.grad(data, target)
        return param

    def train(self, show_every=10):
        reduction_factor = 1
        num_batches = int(np.shape(self.train_data)[0]/self.config.batch_size)
        for epoch in range(self.config.num_epochs):
            if epoch % 50 == 0:
                reduction_factor *= 0.5
            # for batch_label, batch_input in batch_iter(
            #         self.train_labels, self.train_data, self.config.batch_size, num_batches=num_batches):
            #     self.weights = self.sdg(self.weights, batch_input, batch_label)
            for idx in range(num_batches):
                batch_input = self.train_data[idx*self.config.batch_size: (idx + 1)*self.config.batch_size]
                batch_label = self.train_labels[idx*self.config.batch_size: (idx + 1)*self.config.batch_size]
                self.weights = self.sdg(self.weights, batch_input, batch_label,
                                        self.config.learning_rate*reduction_factor)

            self.train_loss = self.loss(self(self.train_data), self.train_labels)
            if epoch % show_every == 0 or epoch == self.config.num_epochs - 1:
                self.train_predictions = self.predict(self(self.train_data))
                correct = np.sum(self.train_predictions == self.train_labels)
                self.train_accuracy = correct / np.shape(self.train_data)[0]
                # if self.accuracy > self.train_accuracy:
                #     self.best_accuracy = self.train_accuracy
                #     self.best_weights = self.weights
                print("Epoch : ", epoch)
                print("Train loss : ", self.train_loss)
                print("Train accuracy : ", self.train_accuracy)
                if self.config.mode == 'cv':
                    self.test()

        # self.weights = self.best_weights

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
        self.weights = pickle.load(open('config/weights' + self.label +'.p', 'rb'))

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
                x, y = randomize_samples(x, y)
                self.classifiers.append(classifier(config, train_set=(x, y)))
        self.classifier_weights = np.ones(num_classifiers)
        self.test_predictions = None
        self.label = label

    def __call__(self, data):
        print(self.classifiers[0](data))
        output = np.zeros(np.shape(data)[0])
        for classifier in self.classifiers:
            output += 1/len(self.classifiers) * classifier(data)
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
            self.accuracy += 1/len(self.classifiers) * classifier.best_accuracy
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
        print("ensemble accuracy " + str(ensemble.accuracy) + 30*"=")
        if ensemble.accuracy > best_accuracy:
            best_accuracy = ensemble.accuracy
            best_lambda = lambda_
        print("best_lambda :", best_lambda)

def ensemble(config, train_set=None, test_set=None, number=1):
    output = 0
    for i in range(number):
        classifier = LogisticClassifier(config, test_set=test_set, label='log_' + str(number))
        classifier.load_weights()
        output += 1/number * classifier(test_set[0])
    y_pred = np.zeros(np.shape(test_set[0])[0])
    y_pred[np.where(output <= 0.5)] = -1
    y_pred[np.where(output > 0.5)] = 1
    return y_pred

def accuracy(y_pred, labels):
    labels[np.where(labels == 0)] = -1
    accuracy = np.sum(y_pred == labels)/np.shape(labels)[0]
    return accuracy


def find_best_lambda(model):
    lambdas = np.logspace(-2, -1.5, 5)
    weights_history = []
    accuracies = []
    train_losses = []
    test_losses = []
    best_weigths = None
    best_accurary = 0
    best_combination = 0
    for idx, lambda_ in enumerate(lambdas):
        model.reset()
        model.config.lambda_ = lambda_
        model.train()
        weights_history.append(model.weights)
        accuracies.append(model.accuracy)
        train_losses.append(model.train_loss)
        test_losses.append(model.test_loss)

        if accuracies[-1] > best_accurary:
            best_accurary = model.accuracy
            best_weigths = model.weights
            best_combination = idx
        print('current best accuracy: ', best_accurary)
        print('current best :', lambdas[idx])

    print('best combination lambda : ', lambdas[best_combination])
    return lambdas, best_weigths, best_accurary, test_losses, train_losses, best_combination


if __name__ == '__main__':
    # find_best_regularizer(EnsembleClassifiers, np.logspace(-3, -2.5, 5))
    x, y = dataloader(mode='train', reduced=False)
    x_test = dataloader(mode='test', reduced = False)
    x = standardize(x)
    x_test = standardize(x_test)
    # # train_dataset, test_dataset = split_data(x, y, ratio=0.9)
    # # train_set = (build_polynomial(train_dataset[0]), train_dataset[1])
    # # test_set = (build_polynomial(test_dataset[0]), test_dataset[1])
    # # # x = dataloader(mode='test', reduced=False)
    # # # x = standardize(x)
    # # # x = build_polynomial(x)
    config = Config(batch_size=200, num_epochs=200, learning_rate=5*10**-4,
                    lambda_= 0.00316227766017,
                    mode='train')
    ensemble = EnsembleClassifiers(config, build_polynomial(x), y, 50, LogisticClassifier,
                                   label='ensemble_50')
    ensemble.train()
    # ensemble.save()
    # ensemble.load_weights()
    output = ensemble(build_polynomial(x_test))
    # # print(output)
    predictions = ensemble.predict(ensemble(build_polynomial(x_test)))
    # # print(predictions)
    create_csv_submission(np.arange(350000, 350000 + x_test.shape[0]), predictions,
                                                                'dataset/submission_01.csv')
    # # y_test[np.where(y_test) == 0] = -1
    #
    # accuracy = np.sum(ensemble.predict(ensemble(build_polynomial(x))) == y)/np.shape(x)[0]
    # print("accuracy loaded weighs", accuracy)



    # model = LogisticClassifier(config, train_set, test_set)
    # find_best_lambda(model)
    # pred = ensemble(config, test_set=test_set, number=4)
    # acc = accuracy(pred, test_set[1])
    # print('accuracy ', acc)
    # create_csv_submission(np.arange(350000, 350000 + x.shape[0]), pred, \
    #                                                             '../dataset/submission_00.csv')

    # log_classifier = LogisticClassifier(config, train_set, test_set, label='log_4')
    # log_classifier.train()
    # log_classifier.save()
    # log_classifier.load_weights()
    # log_classifier.test()
    # ensemble = EnsembleClassifiers(config, train_set, test_set, 5, LogisticClassifier, "ensemble_0")
    # ensemble.train()
    best_lambda = .0133352143216