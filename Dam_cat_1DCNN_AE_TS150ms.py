import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
import copy

data = loadmat('X_FEATURE_snr55_60_65.mat')
X_Feature = data.get('X_FEATURE_snr55_60_65')
idx = np.arange(len(X_Feature))
np.random.shuffle(idx)

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
x_feture_tensor = torch.from_numpy(X_Feature)
x_train = torch.zeros_like(x_feture_tensor)
for i in idx:
    x_train[i] = x_feture_tensor[i]
x_train_tensor = x_train.reshape(len(idx),1,41).float().to(device)   
# convert ndarray into torch tensor
#X_train_tensor = torch.from_numpy(X_train).float().to(device)

# MODEL STRUCTURE
class Conv1D_AE(nn.Module):
    def __init__(self):
        super(Conv1D_AE, self).__init__()
        self.conv1d_encoder=nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4,stride=2),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, 4, 2),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, 4,1),
                nn.BatchNorm1d(128)
        )
        self.fc_encoder = nn.Sequential(
                nn.Linear(5*128,320),
                nn.LeakyReLU(0.1),
                nn.Linear(320,80),
                nn.LeakyReLU(0.1),
                nn.Linear(80, 16),
                nn.LeakyReLU(),
                nn.Linear(16, 8)
            )
        self.fc_decoder=nn.Sequential(
                nn.Linear(8,  16),
                nn.LeakyReLU(),
                nn.Linear(16,80),
                nn.LeakyReLU(),
                nn.Linear(80,320),
                nn.LeakyReLU(),
                nn.Linear(320, 5*128),
                nn.Tanh()
        )
        self.conv1d_decoder = nn.Sequential(
                nn.ConvTranspose1d(128, 64, 4,2),
                nn.BatchNorm1d(64),
                nn.ConvTranspose1d(64, out_channels=32, kernel_size=4, stride=3),
                nn.BatchNorm1d(32),
                nn.ConvTranspose1d(32, 1, 5,1),
                nn.Tanh()
            )
    def forward(self, x):
        x=self.conv1d_encoder(x)
        x = x.view(-1,128*5)
        x = self.fc_encoder(x)
        x = self.fc_decoder(x)
        x = x.view(-1, 128, 5)
        x = self.conv1d_decoder(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features
    
# Now we can create a model and send it at once to the device
AE_model = Conv1D_AE().to(device)
# intial weight of fc_encoder output layer
intial_weight =copy.deepcopy(AE_model.fc_encoder[6].weight)
plt.imshow(intial_weight.detach().numpy())
plt.show()
# TRAINING STEP
batch_size = 120
train_loss_4 = []
train_loss_8 = []
train_loss_16 = []
def train(AE_model, num_epochs, batch_size, lr):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    #optimizer = optim.SGD(AE_model.parameters(), lr=lr,momentum=0.90,weight_decay=0.001)
    optimizer = optim.Adam(AE_model.parameters(),
                                 lr, 
                                 weight_decay=1e-5) 
    train_loader = DataLoader(x_train_tensor, batch_size, shuffle=True)
    outputs = []
    run_loss = []
    #train_loss = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img = data
            recon = AE_model(img)
            loss = criterion(recon, img)
            run_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_8.append(np.sum(loss.item()))
        plt.plot(np.log10(train_loss_8),'r-')
        plt.xlabel('epoch')
        plt.ylabel('log loss')
        if epoch % 40 == 0:
            print('Epoch:{}, Loss:{:.6f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    #return outputs
    plt.show()

lr = 0.005
max_epochs = 15000
outputs = train(AE_model, num_epochs=max_epochs, batch_size= batch_size, lr=lr)
# train weight of fc_encoder output layer
train_weight =copy.deepcopy(AE_model.fc_encoder[6].weight)
plt.imshow(train_weight.detach().numpy())
plt.show()
# weigth update
update_weight = train_weight - intial_weight
plt.imshow(update_weight.detach().numpy())
plt.show()
# test AE_model
def test_model(x,n):
    y_hat=AE_model(x)
    y=x_train_tensor[n][0].numpy()
    y_hat=y_hat[0][0].detach().numpy()
    plt.figure(figsize=(6,4))
    plt.plot(list(range(1,82,2)),y,'g-.',linewidth=1.5)
    plt.plot(list(range(1,82,2)),y_hat,'b-.',linewidth=1.5)
    plt.legend(['original','reconstructed'])
    #plt.savefig(figname)
    plt.show()
"""**************************
n = 1
x = x_train_tensor[n].reshape(1,1,41)
test_model(x,n)
for i, x in enumerate(x_train_tensor):
    if i < 3:
        x = x.view(1,1,41)
        test_model(x, i)
#save and the train_model
#file_model = 'Conv1D_AE.pth' 
#torch.save(Conv1D_AE, file_model) # Save model structure
# file = 'AE_CF_4_snr55_60_65-2.pth'  #trained model file 
# torch.save(AE_model.state_dict(), file)  #saving trained model

# Load model 
#AE_model_trained = Conv1D_AE()
#AE_model_trained.load_state_dict(torch.load(file))
#extacting weight as feature for classifier
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

AE_model.fc_encoder.register_forward_hook(get_activation('fc2'))
x = torch.randn(1, 1, 41)
output = AE_model(x)
print(activation['fc2'])

for j in range(1000):
    for i in x_train_tensor[j]:
        AE_model.fc_encoder.register_forward_hook(get_activation('fc_encoder'))
        x = i.view(1,1,41)
        output = AE_model(x)
        #print(activation['fc2'])
        plt.plot(activation['fc_encoder'][0].numpy())
plt.show()
###############################################################################
################################ PCA ##########################################
###############################################################################

from sklearn.decomposition import PCA
X_Feature = x_train_tensor.numpy().reshape(3600, 41)
pca = PCA(n_components=8, svd_solver='randomized')
X_train_pca = pca.fit_transform(X_Feature)
# Projection
X_pca_comp = pca.components_
X_proj = pca.inverse_transform(X_train_pca)
s_n = 1108
plt.plot(list(range(1,82,2)),X_Feature[s_n]/np.max(X_Feature[s_n]),'g-.')
plt.plot(list(range(1,82,2)),X_proj[s_n]/np.max(X_proj[s_n]),'b-.')
plt.legend(['original', 'reconsturcted'])
plt.show()
"""
 