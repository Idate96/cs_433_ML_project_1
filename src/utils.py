import numpy as np
import csv
np.random.seed(seed=3)

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1

    return y_pred

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

def dataloader(mode='train', reduced=False):
    """Load datasets"""

    print("Loading data ...")
    file_name = 'dataset/' + mode + '.csv'
    with open(file_name) as f:
        first_line = f.readline()
        columns_headears = first_line.split(',')
        indeces_wo_phi = [idx for idx in range(30) if 'phi' not in columns_headears[idx]]

    table = np.genfromtxt(file_name, dtype=float, delimiter=',', skip_header=1,
                          converters={1: lambda x: float(x == b's')}, usecols=indeces_wo_phi)

    features = table[:, 2:]
    labels = table[:, 1]
    print("Data extracted.")
    if mode == 'train':
        return features, labels
    else:
        return features

def randomize_samples(x, y):
    indeces = np.arange(np.shape(x)[0])
    random_indeces = np.random.permutation(indeces)
    return x[random_indeces], y[random_indeces]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def standardize(x):
    """sets mean to 0 and std to 1 approx"""
    x = (x-np.mean(x, axis=0))/(np.std(x, axis=0) + 10**-8)
    return x

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    # np.random.seed(seed)
    num_train_samples = int(np.shape(x)[0]*ratio)
    indeces = np.arange(np.shape(x)[0])
    random_indeces = np.random.permutation(indeces)
    train_idx, test_idx = random_indeces[:num_train_samples], random_indeces[num_train_samples:]
    train_x, train_y = x[train_idx], y[train_idx]
    test_x, test_y = x[test_idx], y[test_idx]
    return (train_x, train_y), (test_x, test_y)

def split_data_k_fold(x, y, begin, k):
    """K fold cross validation.
    Split the data in 10 parts : 9 are used for training the remaining one for cross validation
    To be applied 10 times.
    """
    test_indeces = np.arange(begin*int(np.shape(x)[0]/k), (begin+1)*int(np.shape(x)[0]/k))
    train_indeces = np.asarray(list(range(0, begin*int(np.shape(x)[0]/k))) \
                    + list(range((begin+1)*int(np.shape(x)[0]/k), np.shape(x)[0])))
    train_x, train_y = x[train_indeces], y[train_indeces]
    test_x, test_y = x[test_indeces], y[test_indeces]
    return (train_x, train_y), (test_x, test_y)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_polynomial(x):
    """Builds a non linear base out of the original data"""
    base_mixed = np.zeros((np.shape(x)[0],int(np.shape(x)[1]*(np.shape(x)[1]-1)/2)))

    bias = np.ones(np.shape(x)[0])
    counter = 0

    for i in range(np.shape(x)[1]):
        for j in range(i):
            base_mixed[:, counter] = x[:, i] * x[:, j]
            counter += 1

    base = np.hstack((bias[:, np.newaxis], np.log(abs(1+x)), x, base_mixed, x**2, x**3))
    return base
