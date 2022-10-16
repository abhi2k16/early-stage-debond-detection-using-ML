import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

data = loadmat('X_feature.mat')
target = loadmat('T_reg.mat')

X_Feature = data.get('X_Feature')
X_Feature = X_Feature
target = target.get('T_reg')
target_1 = target
target_1  = target_1.reshape(177)
target_1 = np.array(target_1,dtype='float')

target_1 = target_1/np.max(target_1)
# Targer value for class
T_class = torch.arange(0,177)
for i in range(177):
    if i <= 75:
        T_class[i] = 1
    elif i > 75 and i <= 105:
        T_class[i] = 2
    elif i > 105 and i <= 150:
        T_class[i] = 3
    elif i > 150:
        T_class[i] = 4
# Change the class value into one_hot encoder 
T_class = F.one_hot(T_class % 4)

X_train, X_test, y_train, y_test = train_test_split(X_Feature, T_class, test_size=0.1, random_state=42)

# checking device
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# convert ndarray into torch tensor
X_train_tensor = torch.from_numpy(X_train).float().to(device)
X_test_tensor = torch.from_numpy(X_test).float().to(device)


class nn_model(nn.Module):
    def __init__(self):
        super(nn_model, self).__init__()
        self.fc1 = nn.Linear(41,20)
        self.fc2 = nn.Linear(20,10)
        self.fc3 = nn.Linear(10,4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate a neural network
model = nn_model().to(device)

from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Train The model on training data
from torch.autograd import Variable

# Function to save the model
def saveModel():
    path = './dam_cat_classification.pth'
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loadder:
            inputs, labels = data
            outputs = model(inputs)
            # The labels with highest energy will be our prediction
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (pred == labels).sum().item()
    # Compute the accuracy for all test inputs
    accuracy = (100 * accuracy / total)
    return accuracy
# training function

def train(epochs):
    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader,0):
            inputs = Variable(inputs)
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backwards()
            running_loss += loss.item()

            if i % 10 == 0:
                print('[%d, %5D] loss: %.3f' % (epoch + 1, i+1, running_loss/10))
                running_loss = 0.0
        accuracy = testAccuracy()
        print('For epoch', epoch +1, 'the test accuracy over whole data is %d %%' %(accuracy))

        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy





batch_size = 159
n_itr = 5
batch_size  = batch_size
training_loss = []

for epoch in range(n_itr):
    running_loss = 0
    for i in range(int(X_train_tensor.size()[0]/batch_size)):
        inputs  = torch.index_select(
            X_train_tensor,0,
            torch.linspace(i*batch_size,(i+1)*batch_size - 1, steps= batch_size).long())
        labels = torch.index_select(
            y_train,0,torch.linspace(i*batch_size, (i+1)*batch_size - 1, steps= batch_size).long())
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
    training_loss.append(running_loss/(X_train_tensor.size()[0]/10))
    if epoch % 10 == 0:
        print('At iteration: %d / %d ; Training Loss: %f '%(epoch +1, n_itr, running_loss/(X_train_tensor.size()[0]/10)))
print('Finished Training')
fig = plt.figure()        
plt.plot(range(epoch+1),training_loss,'g-',label='Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.show()

# Test
X_data=X_train_tensor
y_h=model(X_data)
y_pred=y_h.detach().numpy()
y_act=y_train
y_pred_arr=y_pred
y_act_arr=y_act
# Plot test result 
fig = plt.figure()  
plt.plot(y_pred,'g.')
plt.plot(y_act,'b.')
plt.show()


############################### SVR ###########################################
# from sklearn.svm import SVR

# rng = np.random.RandomState(0)
# X = X_train
# y = y_train
# svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
# y_pred_1 = svr_rbf.fit(X_test, y_test).predict(X_test)

# #Plot result
# plt.plot(y_pred_1,'r.-')
# plt.plot(y_test,'b.-')
# plt.show()