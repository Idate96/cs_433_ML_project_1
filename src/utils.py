import numpy as np

def xavier_init(size):
    var = 2/(np.sum(size))
    return var * np.random.randn(*size)

def adam(theta, m, v, beta_1, beta_2, learning_rate,  gradient, iter_num):
    m = (beta_1 * m + (1 - beta_1) * gradient)/(1-beta_1**iter_num)
    v = (beta_2 * v + (1 - beta_2) * gradient**2)/(1-beta_2**iter_num)
    return theta - learning_rate*m/(v**0.5 + 10**-8), m, v


def dataloader(mode='train', reduced=False):
    print("Loading data ...")
    file_name = '../dataset/' + mode + '.csv'
    with open(file_name) as f:
        first_line = f.readline()
        columns_headears = first_line.split(',')
        indeces_wo_phi = [idx for idx in range(30) if 'phi' not in columns_headears[idx]]

    table = np.genfromtxt(file_name, dtype=float, delimiter=',', skip_header=1,
                          converters={1: lambda x: float(x == b's')}, usecols=indeces_wo_phi)

    if reduced:
        features = table[:10000, 2:]
        labels = table[:10000, 1]
    else:
        features = table[:, 2:]
        labels = table[:, 1]
    print("Data extracted.")
    if mode == 'train':
        return features, labels
    else:
        return features

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def standardize(x):
    x = (x-np.mean(x, axis=0))/(np.std(x, axis=0) + 10**-8)
    return x

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    num_train_samples = int(np.shape(x)[0]*ratio)
    indeces = np.arange(np.shape(x)[0])
    random_indeces = np.random.permutation(indeces)
    train_idx, test_idx = random_indeces[:num_train_samples], random_indeces[num_train_samples:]
    train_x, train_y = x[train_idx], y[train_idx]
    test_x, test_y = x[test_idx], y[test_idx]
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
    # for i in range(np.shape(x)[1]):
    #     for j in range(np.shape(x)[1]):
    #         base_mixed_cube[:, counter] = x[:, i]**2 * x[:, j]
    #
    base = np.hstack((bias[:, np.newaxis], x, base_mixed, x**2, x**3))
    return base

if __name__ == '__main__':
    pass