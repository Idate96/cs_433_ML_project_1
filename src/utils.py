import numpy as np



def dataloader(mode= 'train', reduced=False):
    table = np.genfromtxt('dataset/' + mode + '.csv', dtype=float, delimiter=',', skip_header=1,
                          converters={1: lambda x: float(x == b's')})

    if reduced:
        features = table[:10000, 2:]
        labels = table[:10000, 1]
    else:
        features = table[:, 2:]
        labels = table[:, 1]
    if mode == 'train':
        return features, labels
    else:
        return features

def standardize(x):
    x = (x-np.mean(x, axis=0))/(np.std(x, axis=0))
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


if __name__ == '__main__':
    pass