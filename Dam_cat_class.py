import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt


class nn_model(nn.Module):
    def __init__(self):
        super(nn_model, self).__init__()
        self.fc1 = nn.Linear(41, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def saveModel(model):
    path = './dam_cat_classification.pth'
    torch.save(model.state_dict(), path)


def testAccuracy(model, test_loader, loss_fn):
    model.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            _, true = torch.max(labels, 1)
            total += labels.size(0)
            accuracy += (pred == true).sum().item()
    return 100 * accuracy / total


def train(model, optimizer, loss_fn, X_train_tensor, y_train, epochs, batch_size):
    best_accuracy = 0.0
    training_loss = []
    for epoch in range(epochs):
        running_loss = 0
        for i in range(int(X_train_tensor.size(0) / batch_size)):
            inputs = torch.index_select(
                X_train_tensor, 0,
                torch.linspace(i * batch_size, (i + 1) * batch_size - 1, steps=batch_size).long())
            labels = torch.index_select(
                y_train, 0,
                torch.linspace(i * batch_size, (i + 1) * batch_size - 1, steps=batch_size).long())
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        training_loss.append(running_loss / (X_train_tensor.size(0) / 10))
        if epoch % 10 == 0:
            print('At iteration: %d / %d ; Training Loss: %f ' % (
                epoch + 1, epochs, running_loss / (X_train_tensor.size(0) / 10)))
    print('Finished Training')
    return training_loss


if __name__ == '__main__':
    data = loadmat('X_feature.mat')
    target = loadmat('T_reg.mat')

    X_Feature = data.get('X_Feature')
    target = target.get('T_reg')
    target_1 = target.reshape(177)
    target_1 = np.array(target_1, dtype='float')
    target_1 = target_1 / np.max(target_1)

    T_class = torch.arange(0, 177)
    for i in range(177):
        if i <= 75:
            T_class[i] = 1
        elif i > 75 and i <= 105:
            T_class[i] = 2
        elif i > 105 and i <= 150:
            T_class[i] = 3
        elif i > 150:
            T_class[i] = 4
    T_class = F.one_hot(T_class % 4)

    X_train, X_test, y_train, y_test = train_test_split(X_Feature, T_class, test_size=0.1, random_state=42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    model = nn_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    batch_size = 159
    n_itr = 5
    training_loss = train(model, optimizer, loss_fn, X_train_tensor, y_train, n_itr, batch_size)

    fig = plt.figure()
    plt.plot(range(len(training_loss)), training_loss, 'g-', label='Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.show()

    X_data = X_train_tensor
    y_h = model(X_data)
    y_pred = y_h.detach().numpy()
    y_act = y_train

    fig = plt.figure()
    plt.plot(y_pred, 'g.')
    plt.plot(y_act, 'b.')
    plt.show()
