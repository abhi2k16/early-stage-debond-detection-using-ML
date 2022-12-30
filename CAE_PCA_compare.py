from ConvAE_BNL_16 import Conv1D_AE_BNL_16
from ConvAE_BNL_8 import Conv1D_AE_BNL_8
from ConvAE_BNL_4 import Conv1D_AE_BNL_4
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch

# LoadData
data = loadmat('X_FEATURE_snr55_60_65.mat')
X_Feature = data.get('X_FEATURE_snr55_60_65')
x_tensor = torch.from_numpy(X_Feature).view(len(X_Feature), 1, 41).float()
# Load Model

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
#extacting weight as feature for classifier
"""
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
### Plot extracted featuere
for j in range(1000):
    for i in x_tensor[j]:
        AE_model_trained_1.fc_encoder.register_forward_hook(get_activation('fc_encoder'))
        x = i.view(1,1,41)
        output = AE_model_trained_1(x)
        #print(activation['fc2'])
        plt.plot(activation['fc_encoder'][0].numpy())
plt.show()
"""
######################                PCA             #########################       
from sklearn.decomposition import PCA
X_Feature = x_tensor.numpy().reshape(3600, 41)
n_com1 = 4
n_com2 = 15
pca = PCA(n_components=n_com1, svd_solver='randomized')
pca_2 = PCA(n_components=n_com2, svd_solver='randomized')
X_train_pca = pca.fit_transform(X_Feature)
X_train_pca_2 = pca_2.fit_transform(X_Feature)
# Projection
X_proj = pca.inverse_transform(X_train_pca)
X_proj_2 = pca_2.inverse_transform(X_train_pca_2)
"""
# Compare AE and PCA
def comp_AE_PCA(x,n):
    y_hat=AE_model_trained_3(x)
    y=x_tensor[n][0].numpy()
    y_hat=y_hat[0][0].detach().numpy()
    plt.figure(figsize=(6,4))
    plt.plot(list(range(1,82,2)),y,'g-',linewidth=2)
    plt.plot(list(range(1,82,2)),y_hat,'b-.',linewidth=2)
    #plt.plot(list(range(1,82,2)),X_proj_2[n],'b-.',linewidth=2)
    plt.plot(list(range(1,82,2)),X_proj[n],'r-.',linewidth=2)
    plt.legend(['original','AE reconstructed','PCA reconstructed'])
    #plt.savefig(figname)
    plt.show()
n = 1
x = x_tensor[n].reshape(1,1,41)
comp_AE_PCA(x,n)
for i, x in enumerate(x_tensor):
    if i < 2:
        x = x.view(1,1,41)
        comp_AE_PCA(x, i)
"""
# compute mean square difference b/w original and constructed signal
from sklearn.metrics import mean_squared_error as mse
from sklearn.utils import shuffle
X_Train_Pca = np.concatenate((X_train_pca, X_train_pca_2), axis=1)
X_Feature, X_Train_Pca = shuffle(X_Feature, X_Train_Pca)

X_train_pca = X_Train_Pca[:,0:n_com1]
X_train_pca_2 = X_Train_Pca[:,n_com1:n_com1+n_com2]

x_train_tensor = torch.from_numpy(X_Feature).reshape(len(X_Feature),1,41).float()

def MSE(x, y):
    msd = mse(x,y)
    return msd
def cal_mse_pca(x, x_pca):
    y = pca.inverse_transform(x_pca)
    mse_pca = mse(x, y)
    return mse_pca
def cal_mse_pca_2(x, x_pca_2):
    y_2 = pca_2.inverse_transform(x_pca_2)
    mse_pca_2 = mse(x, y_2)
    return mse_pca_2

def cal_mse(model, x):
    y = model(x).view(41).detach().numpy()
    mse_ = MSE(x.view(41).numpy(), y)
    return mse_
# Calculate mse for PCA and AE and compare 
for i in range(1200):
    # mse_sum_1 = np.sum(
    #     cal_mse(model = AE_model_trained_1,x= x_train_tensor[i].view(1,1,41))/2
    #     )
    mse_sum_1 = np.sum(cal_mse_pca_2(x = x_train_tensor[i].numpy().reshape(41,),
                                    x_pca_2 = X_train_pca_2[i]))/2

    mse_sum_2 = np.sum(cal_mse_pca(x = x_train_tensor[i].numpy().reshape(41,),
                                   x_pca = X_train_pca[i]))/2
    plt.plot(i,mse_sum_1,'b.')
    plt.plot(i,mse_sum_2,'r.')
    plt.xlabel('sample number')
    plt.ylabel('Constructution Error')
    plt.legend(['AE','PCA'])
    plt.ylim((0,0.007))
plt.show()
# Plot average mse for PCA and AE
avg_ae_16 = []
avg_pca_16 = []
avg_ae_8 = []
avg_pca_8 = []
avg_ae_4= []
avg_pca_4 = []
for i in range(3600):
    err = np.sum(cal_mse_pca_2(x = x_train_tensor[i].numpy().reshape(41,),
                                    x_pca_2 = X_train_pca_2[i]))/2
    avg_ae_4.append(err)
    err_2 =  np.sum(cal_mse_pca(x = x_train_tensor[i].numpy().reshape(41,),
                                   x_pca = X_train_pca[i]))/2
    avg_pca_4.append(err_2)
AVG_AE_16 = np.sum(np.array(avg_ae_16))
Avg_pca_16 = np.sum(np.array(avg_pca_16))
AVG_AE_8= np.sum(np.array(avg_ae_8))
Avg_pca_8 = np.sum(np.array(avg_pca_8))
AVG_AE_4 = np.sum(np.array(avg_ae_4))
Avg_pca_4 = np.sum(np.array(avg_pca_4))

AVG_ae = np.array([ 1.1543722874339437,1.1540957312136015,1.046295133382955])/3600
AVG_pca = np.array([ 3.3298686648340663,2.157393283203419,1.1139823185709247])/3600
n_com =np.array([4,8,16])

plt.plot(n_com, AVG_ae,'b-*')
plt.plot(n_com, AVG_pca,'r-*')
plt.xlabel('Number of Component')
plt.ylabel('Average Cons. Error')
plt.legend(['AE','PCA'])
plt.xticks([4,8,16])
plt.show()

    

"""
# Plot pca varience ratio
pca_val = pca.explained_variance_ratio_
pca_cum = np.cumsum(pca_val)
plt.bar(range(1,len(pca_val)+1), pca_val, width=0.9, align='center', label='Individual explained variance')
plt.step(range(1,len(pca_cum)+1), pca_cum, where='mid',label='Cumulative explained variance')
for i, v in enumerate(pca_cum):
    plt.text(i+0.5, v + 0.02, '{0:.2f}'.format(v))
plt.ylim((0, 1))
plt.xlim((0,17))
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='center right')
plt.tight_layout()
plt.show()
"""

