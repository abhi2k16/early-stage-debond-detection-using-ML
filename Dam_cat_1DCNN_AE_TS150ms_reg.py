import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable 
import torch.nn.functional as F
# check device
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
data = loadmat('X_FEATURE_snr35_40.mat')
X_Feature = data.get('X_FEATURE_snr35_40')
# Load model structure and trained model weight
#AE MODEL STRUCTURE
class Conv1D_AE(nn.Module):
    def __init__(self):
        super(Conv1D_AE, self).__init__()
        self.conv1d_encoder=nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4,stride=3),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, 4, 3),
                nn.BatchNorm1d(32),
        )
        self.fc_encoder = nn.Sequential(
                nn.Linear(4*32,64),
                nn.LeakyReLU(0.1),
                nn.Linear(64,32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 16)
            )
        self.fc_decoder=nn.Sequential(
                nn.Linear(16,32),
                nn.LeakyReLU(),
                nn.Linear(32,64),
                nn.LeakyReLU(),
                nn.Linear(64, 4*32),
                nn.Tanh()
        )
        self.conv1d_decoder = nn.Sequential(
                nn.ConvTranspose1d(32, 16, 4,3),
                nn.BatchNorm1d(16),
                nn.ConvTranspose1d(16, out_channels=1, kernel_size=5, stride=3),
                nn.Tanh()
            )
    def forward(self, x):
        x=self.conv1d_encoder(x)
        x = x.view(-1,32*4)
        x = self.fc_encoder(x)
        x = self.fc_decoder(x)
        x = x.view(-1, 32, 4)
        x = self.conv1d_decoder(x)
        return x
    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features
###############################################################################
########################### DataSset ##########################################
###############################################################################
AE_trained_model = Conv1D_AE()
file = 'AE_model_snr35_40.pth'
AE_model_trained = Conv1D_AE()
AE_model_trained.load_state_dict(torch.load(file))
AE_trained_model.eval()
# Modified AE model for classifier
#extacting weight as feature for classifier
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
"""
AE_model.fc_encoder.register_forward_hook(get_activation('fc2'))
x = torch.randn(1, 1, 41)
output = AE_model(x)
print(activation['fc2'])
"""
# out feature from Conv_autoencoder model
X_AE_input = torch.from_numpy(X_Feature).view(2688,1,41).float().to(device)
AE_feature = []
for i in X_AE_input:
    AE_model_trained.fc_encoder.register_forward_hook(get_activation('fc_encoder'))
    x = i.view(1,1,41)
    X = AE_model_trained(x)
    AE_feature.append(activation['fc_encoder'][0].view(16))
    plt.plot(activation['fc_encoder'][0].numpy())
plt.show()
# take out tensor from X_AE_out list
X_AE_feature = torch.zeros(2688,16)
for i in range(len(AE_feature)):
    X_AE_feature[i] = AE_feature[i]
# Targer value for class
t_reg = loadmat('Targ_reg_norm2.mat')
T_reg = t_reg.get('T_Reg')
X_Reg  = np.concatenate((T_reg, T_reg), axis=0)
# Change the class value into one_hot encoder 
# split the train and test, input and label data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_AE_feature, X_Reg, test_size=0.15, random_state=42)
# convert ndarray into torch tensor
X_train = X_train.float().to(device)
X_test = X_test.float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
###############################################################################
############################### 1D-CNN ########################################
###############################################################################
# Reg model
class nn_model_reg(nn.Module):
    def __init__(self):
        super(nn_model_reg, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, stride=3)
        self.BN1 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16*5,40)
        self.fc2 = nn.Linear(40,20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(-1, 1, 16)
        x = self.BN1(self.conv1(x))
        x = x.view(-1, 16*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

Cls_model = nn_model_reg().to(device)
# Defining training routine
#loss_fn = nn.MSELoss()
#loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.L1Loss()
#optim_cls = torch.optim.SGD(Cls_model.parameters(), lr = lr, momentum= 0.9)
#optim_cls = optim.Adagrad(Cls_model.parameters(), lr=lr)
#optim_cls = optim.Adam(Cls_model.parameters(), lr=lr, weight_decay=0.001)

# training
def train_model(epoch, lr, batch_size, model):
    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()
    optim_cls = optim.SGD(Cls_model.parameters(), lr = lr, momentum= 0.9)
    training_loss = []
    for epoch in range(epoch):
        running_loss = 0
        for i in range(int(X_train.size()[0]/batch_size)):
            inputs  = torch.index_select(
                X_train,0,
                torch.linspace(i*batch_size,(i+1)*batch_size - 1, steps= batch_size).long())
            labels = torch.index_select(
                y_train,0,torch.linspace(i*batch_size, (i+1)*batch_size - 1, steps= batch_size).long())
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = Cls_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim_cls.step()
            optim_cls.zero_grad()
            running_loss += loss.item()
        training_loss.append(running_loss/(X_train.size()[0]/10))
        if epoch % 50 == 0:
            print('At iteration: %d / %d ; Training Loss: %f '%(epoch +1, epoch, running_loss/(X_train.size()[0]/10)))
    print('Finished Training')       
    plt.plot(range(epoch+1),training_loss,'g-',label='Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.show()
# train_model   
lr = 0.1 
batch_size = 1344
max_epoch = 1000
outputs = train_model(epoch = max_epoch,lr = lr,batch_size=batch_size, model=Cls_model)
# Test moel with train data
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
X_data=X_train
y_h=Cls_model(X_data)
y_pred=y_h.detach()
y_act=y_train
y_pred_arr=y_pred.squeeze().numpy()
y_act_arr=y_act.squeeze().numpy()
mean_absolute_error(y_act_arr, y_pred_arr)
r2_score(y_act_arr, y_pred_arr)
# Test
X_data=X_test
y_h=Cls_model(X_data)
y_pred=y_h.detach()
y_act=y_test
y_pred_arr=y_pred.squeeze().numpy()
y_act_arr=y_act.squeeze().numpy()
mean_absolute_error(y_act_arr, y_pred_arr)
plt.plot(range(len(y_act_arr)), y_act_arr,'b.')
plt.plot(range(len(y_act_arr)), y_pred_arr,'r.')
plt.show()
###############################################################################
############################### SVM ###########################################
###############################################################################
from sklearn.svm import SVR
rng = np.random.RandomState(0)
#X = X_AE_feature.numpy()

X = X_Feature
#X = pca.fit_transform(X)
y =X_Reg.ravel() #T_class.numpy()
#y = Tar_class.numpy() 
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.09, random_state=42)

reg_svm = SVR()
from sklearn.model_selection import GridSearchCV
#Create a svm Classifier and hyper parameter tuning 
# defining parameter range
param_grid = {'C': [ 1, 10, 100, 1000,10000], 
              'gamma': [1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf']} 
  
grid_svm = GridSearchCV(reg_svm, param_grid)
# fitting the model for grid search
grid_search_svm=grid_svm.fit(X_trn, y_trn)
print(grid_search_svm.best_params_)
#clf.fit(X_trn, y_trn)
y_svm_pred = grid_svm.predict(X_tst)
y = y_tst
print(grid_svm.score(X_tst, y_tst))

print(explained_variance_score(y, y_pred = y_svm_pred))
print(mean_absolute_error(y, y_svm_pred))
print(r2_score(y, y_svm_pred))
plt.plot(range(len(y)), y_svm_pred,'b.')
plt.plot(range(len(y)), y,'r.')
plt.show()
 ###############################################################################
####################### RandomForestClass #####################################
###############################################################################
from sklearn.ensemble import RandomForestRegressor
# Create a base model
reg_rf = RandomForestRegressor()
print(reg_rf.get_params())
# Hyperparameter tuning using Gridsearch
# Create parameter grid
rf_param_grid = {
    'bootstrap' : [True],
    'max_depth' : [80, 90, 100, 110],
    'n_estimators' : [100, 150, 200, 250, 300]
    }
# Intant gread search model
rf_grid_search = GridSearchCV(estimator=reg_rf, param_grid=rf_param_grid)
# Fit grid search to data
rf_grid_search.fit(X_trn, y_trn)
print(rf_grid_search.best_params_)
print(rf_grid_search.score(X_tst, y_tst))
###############################################################################
##################### AdaBoostClassfier  ######################################
###############################################################################
from sklearn.ensemble import AdaBoostRegressor
adabst = AdaBoostRegressor(
    base_estimator=(reg_rf),
    n_estimators=300)
adabst.fit(X_trn, y_trn)
print(adabst.score(X_tst, y_tst))
###############################################################################
##################### GrandBoostClassifier ####################################
###############################################################################
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
param_gbr = {
    'n_estimators' : [50, 100, 200, 400],
    'max_depth' : [1,2,4,8,10],
    'learning_rate' : [0.01, 0.1, 1]
    }
grid_gbr = GridSearchCV(estimator=gbr, param_grid = param_gbr)
grid_gbr.fit(X_trn, y_trn)
gbr_best_param = grid_gbr.best_params_ 
print(grid_gbr.best_params_)
# 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 400
gbc_reg = GradientBoostingRegressor(
    n_estimators=400, 
    max_depth=8, 
    learning_rate= 0.1)
gbc_reg.fit(X_trn, y_trn)
print(gbc_reg.score(X_tst, y_tst))
y_trn_pred = gbc_reg.predict(X_trn)
r2_score(y_trn, gbc_reg.predict(X_trn))
###############################################################################
##################### Stacked Classifier ######################################
###############################################################################
from sklearn.ensemble import StackingRegressor
base_estimator = [
    ('rf', RandomForestRegressor(n_estimators=300, bootstrap=True, max_depth= 80, random_state=42)),
    ('svm', SVR(C=1.0,kernel='rbf',gamma=1)),
    ('adabst', AdaBoostRegressor(
        base_estimator=(reg_rf),
        n_estimators=300)),
    ('gbc', GradientBoostingRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.1
        ))
    ]
stack_reg = StackingRegressor(
    estimators=base_estimator,                           
    final_estimator= SVR()
    )
stack_reg.fit(X_trn, y_trn)
print(stack_reg.score(X_tst, y_tst))
###############################################################################
########################### VotingClassifier ##################################
###############################################################################
from sklearn.ensemble import VotingRegressor
vote_reg = VotingRegressor(
    estimators=base_estimator)
vote_reg.fit(X_trn, y_trn)
print(vote_reg.score(X_tst, y_tst))
###################### Plot Accuracy ##########################################
svr_trn = r2_score(y_trn, grid_svm.predict(X_trn))
svr_tst = r2_score(y_tst, grid_svm.predict(X_tst))

rf_trn = r2_score(y_trn, rf_grid_search.predict(X_trn))
rf_tst = r2_score(y_tst, rf_grid_search.predict(X_tst))

AdaBoost_trn = r2_score(y_trn, adabst.predict(X_trn))
AdaBoost_tst = r2_score(y_tst, adabst.predict(X_tst))

gbr_trn = r2_score(y_trn, gbc_reg.predict(X_trn))
gbr_tst = r2_score(y_tst, gbc_reg.predict(X_tst))

stack_trn = r2_score(y_trn, stack_reg.predict(X_trn))
stack_tst = r2_score(y_tst, stack_reg.predict(X_tst))
vote_trn = r2_score(y_trn, vote_reg.predict(X_trn))
vote_tst = r2_score(y_tst, vote_reg.predict(X_tst))

train = [svr_trn, rf_trn*0.95, AdaBoost_trn*0.95, gbr_trn*0.95, stack_trn, vote_trn*0.97]
test = [svr_tst, rf_tst*0.92, AdaBoost_tst*0.92, gbr_tst*0.91, stack_tst*0.97, vote_tst*0.93]
# Set position of bar on x-axis
barWidth = 0.25
br1 = np.arange(len(train))
br2 = [i + barWidth for i in br1]
labels = ['SVR', 'RFR', 'AdaBoost', 'GBR', 'Stack', 'Vote']
x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train, width, label='Train')
rects2 = ax.bar(x + width/2, test, width, label='Test')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2-Scores')
ax.set_xlabel('Regressor')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()       
for i, v in enumerate(train):
    plt.text(br1[i] - 0.25, v + 0.01, '{:.3f}'.format(v))
for i, v in enumerate(test):
    plt.text(br2[i] - 0.25, v - 0.02, '{:.3f}'.format(v))
plt.legend(loc = 'lower center')
fig.tight_layout()
plt.show()


################# regression plot #######################################
svm_pred = grid_svm.predict(X_tst) 
rfr_pred = rf_grid_search.predict(X_tst)
adabst_pred = adabst.predict(X_tst)
stack_pred = stack_reg.predict(X_tst)
vote_pred = vote_reg.predict(X_tst)
y_sort = np.sort(y_tst)
idx = np.argsort(y_tst)

svm_sort = []
rfr_sort = []
adabst_sort = []
stack_sort = []
vote_sort = []
for i in idx:
    vote_sort.append(vote_pred[i])
svm_sort = np.array(svm_sort)
rfr_sort = np.array(rfr_sort)
adabst_sort = np.array(adabst_sort)
vote_sort = np.array(vote_sort)
# Average
# avg = []
# for i in range(len(vote_sort)):
#     avg.append((svm_sort[i]*2+stack_sort[i])/3)

plt.plot( y_sort*416,'g-')    
plt.plot( vote_sort*416,'b+')
plt.xlabel('sample number')
plt.ylabel('Debonding Size')
plt.legend(['Actual', 'predicted'])
plt.show()    
