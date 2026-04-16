from ConvAE_BNL_16 import Conv1D_AE_BNL_16
from ConvAE_BNL_8 import Conv1D_AE_BNL_8
from ConvAE_BNL_4 import Conv1D_AE_BNL_4
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.utils import shuffle
import torch


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def MSE(x, y):
    return mse(x, y)


def cal_mse_pca(pca, x, x_pca):
    y = pca.inverse_transform(x_pca)
    return mse(x, y)


def cal_mse_pca_2(pca_2, x, x_pca_2):
    y_2 = pca_2.inverse_transform(x_pca_2)
    return mse(x, y_2)


def cal_mse(model, x):
    y = model(x).view(41).detach().numpy()
    return MSE(x.view(41).numpy(), y)


if __name__ == '__main__':
    data = loadmat('X_FEATURE_snr55_60_65.mat')
    X_Feature = data.get('X_FEATURE_snr55_60_65')
    x_tensor = torch.from_numpy(X_Feature).view(len(X_Feature), 1, 41).float()

    file_1 = 'AE_CF_4_snr55_60_65-2.pth'
    file_2 = 'AE_CF_8_snr55_60_65-3.pth'
    file_3 = 'AE_CF_16_snr55_60_65-2.pth'
    AE_model_trained_1 = Conv1D_AE_BNL_4()
    AE_model_trained_2 = Conv1D_AE_BNL_8()
    AE_model_trained_3 = Conv1D_AE_BNL_16()
    AE_model_trained_1.load_state_dict(torch.load(file_1))
    AE_model_trained_2.load_state_dict(torch.load(file_2))
    AE_model_trained_3.load_state_dict(torch.load(file_3))
    AE_model_trained_1.eval()
    AE_model_trained_2.eval()
    AE_model_trained_3.eval()

    X_Feature = x_tensor.numpy().reshape(3600, 41)
    n_com1 = 4
    n_com2 = 15
    pca = PCA(n_components=n_com1, svd_solver='randomized')
    pca_2 = PCA(n_components=n_com2, svd_solver='randomized')
    X_train_pca = pca.fit_transform(X_Feature)
    X_train_pca_2 = pca_2.fit_transform(X_Feature)
    X_proj = pca.inverse_transform(X_train_pca)
    X_proj_2 = pca_2.inverse_transform(X_train_pca_2)

    X_Train_Pca = np.concatenate((X_train_pca, X_train_pca_2), axis=1)
    X_Feature, X_Train_Pca = shuffle(X_Feature, X_Train_Pca)
    X_train_pca = X_Train_Pca[:, 0:n_com1]
    X_train_pca_2 = X_Train_Pca[:, n_com1:n_com1 + n_com2]
    x_train_tensor = torch.from_numpy(X_Feature).reshape(len(X_Feature), 1, 41).float()

    # MSE comparison plot (first 1200 samples)
    for i in range(1200):
        mse_sum_1 = np.sum(cal_mse_pca_2(pca_2, x_train_tensor[i].numpy().reshape(41,), X_train_pca_2[i])) / 2
        mse_sum_2 = np.sum(cal_mse_pca(pca, x_train_tensor[i].numpy().reshape(41,), X_train_pca[i])) / 2
        plt.plot(i, mse_sum_1, 'b.')
        plt.plot(i, mse_sum_2, 'r.')
    plt.xlabel('sample number')
    plt.ylabel('Construction Error')
    plt.legend(['AE', 'PCA'])
    plt.ylim((0, 0.007))
    plt.show()

    # Average MSE over all 3600 samples
    avg_ae_4, avg_pca_4 = [], []
    for i in range(3600):
        err = np.sum(cal_mse_pca_2(pca_2, x_train_tensor[i].numpy().reshape(41,), X_train_pca_2[i])) / 2
        avg_ae_4.append(err)
        err_2 = np.sum(cal_mse_pca(pca, x_train_tensor[i].numpy().reshape(41,), X_train_pca[i])) / 2
        avg_pca_4.append(err_2)

    AVG_ae = np.array([1.1543722874339437, 1.1540957312136015, 1.046295133382955]) / 3600
    AVG_pca = np.array([3.3298686648340663, 2.157393283203419, 1.1139823185709247]) / 3600
    n_com = np.array([4, 8, 16])

    plt.plot(n_com, AVG_ae, 'b-*')
    plt.plot(n_com, AVG_pca, 'r-*')
    plt.xlabel('Number of Component')
    plt.ylabel('Average Cons. Error')
    plt.legend(['AE', 'PCA'])
    plt.xticks([4, 8, 16])
    plt.show()
