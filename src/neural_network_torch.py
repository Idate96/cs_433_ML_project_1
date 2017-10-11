from torch import optim
import torch
import torch.utils.data
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as f
from utils import dataloader, standardize, split_data

x, y = dataloader(mode='train', reduced=False)
x = standardize(x)
train_dataset, test_dataset = split_data(x, y, ratio=0.9)
test_data, test_target = test_dataset
train_data, train_target = train_dataset


train = torch.utils.data.TensorDataset(torch.from_numpy(train_data).type(torch.FloatTensor),
                                       torch.from_numpy(train_target).type(torch.LongTensor))
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test = torch.utils.data.TensorDataset(torch.from_numpy(test_data).type(torch.FloatTensor),
                                      torch.from_numpy(test_target).type(torch.LongTensor))
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

class SimpleNN(torch.nn.Module):
    def __init__(self, batch_size=128, learning_rate=10**-4, num_epochs= 10, load_weights=False):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.load_weights = load_weights
        self.num_epochs = num_epochs
        # architecture
        self.fc_1 = nn.Linear(30, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 2)

    def forward(self, x):
        x = f.relu(self.fc_1(x))
        x = f.dropout(x)
        x = f.relu(self.fc_2(x))
        x = f.dropout(x)
        x = f.tanh(self.fc_3(x))
        return x

    def init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=10**-3)

    def loss(self, outputs, targets):
        self.loss_func = nn.CrossEntropyLoss()
        loss = self.loss_func(outputs, targets)
        return loss

def new_train(model, epoch):
    log_interval = 250
    model.init_optimizer()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        # target = preprocess_target(target)
        model.optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, target)
        train_loss += loss
        loss.backward()
        model.optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    print("Train loss :", 100*train_loss/np.shape(train_data)[0])


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += model.loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({})\n'.format(
        correct, np.shape(test_data)[0],
        100. * correct / np.shape(test_data)[0]))
    print("Test loss : {}" .format(100*test_loss[0]/np.shape(test_data)[0]))

if __name__ == '__main__':
    net = SimpleNN()
    num_epochs = 100
    for epoch in range(num_epochs):
        new_train(net, epoch)
        test(net)