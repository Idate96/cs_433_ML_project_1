from src.utils import sigmoid, batch_iter, dataloader, split_data,\
    standardize, xavier_init, build_polynomial
import numpy as np
import pickle
import csv

class Config(object):
    """Configuration object for the classifiers"""
    def __init__(self, batch_size, num_epochs, learning_rate, lambda_):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.lambda_ = lambda_


class LogisticClassifier(object):
    def __init__(self, config, train_set, test_set, label='0'):
        self.config = config
        # construct non linear features
        self.train_data, self.train_labels = train_set
        self.test_data, self.test_labels = test_set
        # self.weights = xavier_init(np.shape(self.train_data))
        self.weights = np.zeros((np.shape(self.train_data)[1]))
        self.train_losses = []
        self.test_losses = []
        self.accuracy = 0
        self.test_predictions = None
        self.label = label

    def __call__(self, input):
        return sigmoid(input @ self.weights)

    def loss(self, output, target, sample_weights=1):
        loss = 1 / np.shape(target)[0] * (target - output).T\
               @ (sample_weights * (target - output)) + self.config.lambda_ * self.weights.T @ \
                                                  self.weights
        return loss

    def grad(self, data_batch, target_batch, sample_weights=1):
        return data_batch.T @ ((self(data_batch) - target_batch) * sample_weights *
                self(data_batch) * (1 - self(data_batch))) + \
               self.config.lambda_ * self.weights

    def sdg(self, param, data, target):
        param -= self.config.learning_rate * self.grad(data, target)
        return param

    def train(self, show_every=10):
        num_batches = int(np.shape(self.train_data)[0]/self.config.batch_size)
        for epoch in range(self.config.num_epochs):
            if epoch % 50 == 0:
                self.config.learning_rate *= 0.5
            for batch_label, batch_input in batch_iter(
                    self.train_labels, self.train_data, self.config.batch_size, num_batches=num_batches):
                self.weights = self.sdg(self.weights, batch_input, batch_label)
            train_loss = self.loss(self(self.train_data), self.train_labels)
            if epoch % show_every == 0 or epoch == self.config.num_epochs - 1:
                print("Epoch : ", epoch)
                print("Train loss : ", train_loss)
                self.test()

    def test(self):
        output = self(self.test_data)
        test_loss = self.loss(output, self.test_labels)
        self.test_losses.append(test_loss)
        self.test_predictions = self.predict(output)
        correct = np.sum(self.test_predictions == self.test_labels)
        self.accuracy = correct / np.shape(self.test_data)[0]
        print("Test loss :", test_loss)
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
    def __init__(self, config, train_data, test_data, num_classifiers, classifier, label='0'):
        self.train_data, self.train_labels = train_set
        self.test_data, self.test_labels = test_set
        self.classifiers = [classifier(config, (self.train_data, self.train_labels),
                                               (self.test_data, self.test_labels))
                            for i in range(num_classifiers)]
        self.classifier_weights = np.ones(num_classifiers)
        self.test_predictions = None
        self.label = label

    def check_weights(self):
        for classifier in self.classifiers:
            print(classifier.weights)

    def train(self):
        for classifier in self.classifiers:
            classifier.train()
            self.check_weights()
        self.test()

    def test(self):
        output = 0
        for classifier in self.classifiers:
            output += 1/len(self.classifiers) * classifier(self.test_data)
        self.test_predictions = output > 0.5
        correct = np.sum(self.test_predictions == self.test_labels)
        self.accuracy = correct / np.shape(self.test_data)[0]
        print('Test ensemble accuracy :', self.accuracy)

    def save(self):
        weights = np.zeros((len(self.classifiers), np.shape(self.train_data)[1]))
        for i, classifier in enumerate(self.classifiers):
            weights[i] = classifier.weights
        with open(r'config/' + self.label, "wb") as file:
            pickle.dump(weights, file)

    def load_weights(self):
        weights = pickle.load('config/' + self.label, "wb")
        for i, classifier in enumerate(self.classifiers):
            classifier.weights = weights[i]

def load_classifiers(config, train_set, test_set, number):
    output = 0
    for i in range(number):
        classifier = LogisticClassifier(config, train_set, test_set, label = 'log_' + str(number))
        classifier.load_weights()
        output += 1/number * classifier(test_set[0])
    predictions = output > 0.5
    correct = np.sum(predictions == test_set[1])
    accuracy = correct / np.shape(test_set[0])[0]
    print('Test accuracy :', accuracy)






if __name__ == '__main__':
    x, y = dataloader(mode='train', reduced=False)
    x = standardize(x)
    train_dataset, test_dataset = split_data(x, y, ratio=0.9)
    train_set = (build_polynomial(train_dataset[0]), train_dataset[1])
    test_set = (build_polynomial(test_dataset[0]), test_dataset[1])
    config = Config(batch_size=200, num_epochs=100, learning_rate=5*10**-4, lambda_=0.01)
    load_classifiers(config, train_set, test_set, 4)
    # log_classifier = LogisticClassifier(config, train_set, test_set, label='log_4')
    # log_classifier.train()
    # log_classifier.save()
    # log_classifier.load_weights()
    # log_classifier.test()
    # ensemble = EnsembleClassifiers(config, train_set, test_set, 5, LogisticClassifier, "ensemble_0")
    # ensemble.train()