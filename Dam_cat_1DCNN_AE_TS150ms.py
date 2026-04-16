import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import optim
import copy


class Conv1D_AE(nn.Module):
    def __init__(self):
        super(Conv1D_AE, self).__init__()
        self.conv1d_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 4, 2),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 4, 1),
            nn.BatchNorm1d(128)
        )
        self.fc_encoder = nn.Sequential(
            nn.Linear(5 * 128, 320),
            nn.LeakyReLU(0.1),
            nn.Linear(320, 80),
            nn.LeakyReLU(0.1),
            nn.Linear(80, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8)
        )
        self.fc_decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 320),
            nn.LeakyReLU(),
            nn.Linear(320, 5 * 128),
            nn.Tanh()
        )
        self.conv1d_decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, 2),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, out_channels=32, kernel_size=4, stride=3),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, 5, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1d_encoder(x)
        x = x.view(-1, 128 * 5)
        x = self.fc_encoder(x)
        x = self.fc_decoder(x)
        x = x.view(-1, 128, 5)
        x = self.conv1d_decoder(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(AE_model, x_train_tensor, num_epochs, batch_size, lr):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(AE_model.parameters(), lr=lr, weight_decay=1e-5)
    train_loader = DataLoader(x_train_tensor, batch_size, shuffle=True)
    train_loss = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img = data
            recon = AE_model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss.append(np.sum(loss.item()))
        plt.plot(np.log10(train_loss), 'r-')
        plt.xlabel('epoch')
        plt.ylabel('log loss')
        if epoch % 40 == 0:
            print('Epoch:{}, Loss:{:.6f}'.format(epoch + 1, float(loss)))
    plt.show()


def test_model(AE_model, x_train_tensor, x, n):
    y_hat = AE_model(x)
    y = x_train_tensor[n][0].numpy()
    y_hat = y_hat[0][0].detach().numpy()
    plt.figure(figsize=(6, 4))
    plt.plot(list(range(1, 82, 2)), y, 'g-.', linewidth=1.5)
    plt.plot(list(range(1, 82, 2)), y_hat, 'b-.', linewidth=1.5)
    plt.legend(['original', 'reconstructed'])
    plt.show()


if __name__ == '__main__':
    data = loadmat('X_FEATURE_snr55_60_65.mat')
    X_Feature = data.get('X_FEATURE_snr55_60_65')
    idx = np.arange(len(X_Feature))
    np.random.shuffle(idx)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    x_feture_tensor = torch.from_numpy(X_Feature)
    x_train = torch.zeros_like(x_feture_tensor)
    for i in idx:
        x_train[i] = x_feture_tensor[i]
    x_train_tensor = x_train.reshape(len(idx), 1, 41).float().to(device)

    AE_model = Conv1D_AE().to(device)

    intial_weight = copy.deepcopy(AE_model.fc_encoder[6].weight)
    plt.imshow(intial_weight.detach().numpy())
    plt.show()

    lr = 0.005
    max_epochs = 15000
    batch_size = 120
    train(AE_model, x_train_tensor, num_epochs=max_epochs, batch_size=batch_size, lr=lr)

    train_weight = copy.deepcopy(AE_model.fc_encoder[6].weight)
    plt.imshow(train_weight.detach().numpy())
    plt.show()

    update_weight = train_weight - intial_weight
    plt.imshow(update_weight.detach().numpy())
    plt.show()

    n = 1
    x = x_train_tensor[n].reshape(1, 1, 41)
    test_model(AE_model, x_train_tensor, x, n)
