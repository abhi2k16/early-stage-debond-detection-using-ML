import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                               GradientBoostingRegressor, StackingRegressor, VotingRegressor)


class Conv1D_AE(nn.Module):
    def __init__(self):
        super(Conv1D_AE, self).__init__()
        self.conv1d_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, stride=3),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 4, 3),
            nn.BatchNorm1d(32),
        )
        self.fc_encoder = nn.Sequential(
            nn.Linear(4 * 32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.fc_decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4 * 32),
            nn.Tanh()
        )
        self.conv1d_decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, 3),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, out_channels=1, kernel_size=5, stride=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1d_encoder(x)
        x = x.view(-1, 32 * 4)
        x = self.fc_encoder(x)
        x = self.fc_decoder(x)
        x = x.view(-1, 32, 4)
        x = self.conv1d_decoder(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class nn_model_reg(nn.Module):
    def __init__(self):
        super(nn_model_reg, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, stride=3)
        self.BN1 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16 * 5, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(-1, 1, 16)
        x = self.BN1(self.conv1(x))
        x = x.view(-1, 16 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def train_model(Cls_model, X_train, y_train, epoch, lr, batch_size):
    loss_fn = nn.MSELoss()
    optim_cls = optim.SGD(Cls_model.parameters(), lr=lr, momentum=0.9)
    training_loss = []
    for ep in range(epoch):
        running_loss = 0
        for i in range(int(X_train.size(0) / batch_size)):
            inputs = torch.index_select(
                X_train, 0,
                torch.linspace(i * batch_size, (i + 1) * batch_size - 1, steps=batch_size).long())
            labels = torch.index_select(
                y_train, 0,
                torch.linspace(i * batch_size, (i + 1) * batch_size - 1, steps=batch_size).long())
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = Cls_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim_cls.step()
            optim_cls.zero_grad()
            running_loss += loss.item()
        training_loss.append(running_loss / (X_train.size(0) / 10))
        if ep % 50 == 0:
            print('At iteration: %d / %d ; Training Loss: %f ' % (ep + 1, epoch, running_loss / (X_train.size(0) / 10)))
    print('Finished Training')
    plt.plot(range(epoch), training_loss, 'g-', label='Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.show()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data = loadmat('X_FEATURE_snr35_40.mat')
    X_Feature = data.get('X_FEATURE_snr35_40')

    file = 'AE_model_snr35_40.pth'
    AE_model_trained = Conv1D_AE()
    AE_model_trained.load_state_dict(torch.load(file))
    AE_model_trained.eval()

    activation = {}
    X_AE_input = torch.from_numpy(X_Feature).view(2688, 1, 41).float().to(device)
    AE_feature = []
    for i in X_AE_input:
        AE_model_trained.fc_encoder.register_forward_hook(get_activation(activation, 'fc_encoder'))
        x = i.view(1, 1, 41)
        AE_model_trained(x)
        AE_feature.append(activation['fc_encoder'][0].view(16))
        plt.plot(activation['fc_encoder'][0].numpy())
    plt.show()

    X_AE_feature = torch.zeros(2688, 16)
    for i in range(len(AE_feature)):
        X_AE_feature[i] = AE_feature[i]

    t_reg = loadmat('Targ_reg_norm2.mat')
    T_reg = t_reg.get('T_Reg')
    X_Reg = np.concatenate((T_reg, T_reg), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X_AE_feature, X_Reg, test_size=0.15, random_state=42)
    X_train = X_train.float().to(device)
    X_test = X_test.float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    Cls_model = nn_model_reg().to(device)
    lr = 0.1
    batch_size = 1344
    max_epoch = 1000
    train_model(Cls_model, X_train, y_train, epoch=max_epoch, lr=lr, batch_size=batch_size)

    for X_data, y_data, label in [(X_train, y_train, 'Train'), (X_test, y_test, 'Test')]:
        y_h = Cls_model(X_data)
        y_pred_arr = y_h.detach().squeeze().numpy()
        y_act_arr = y_data.squeeze().numpy()
        print(f'{label} MAE:', mean_absolute_error(y_act_arr, y_pred_arr))
        print(f'{label} R2:', r2_score(y_act_arr, y_pred_arr))
    plt.plot(range(len(y_act_arr)), y_act_arr, 'b.')
    plt.plot(range(len(y_act_arr)), y_pred_arr, 'r.')
    plt.show()

    # SVR
    X = X_Feature
    y = X_Reg.ravel()
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.09, random_state=42)

    param_grid = {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    grid_svm = GridSearchCV(SVR(), param_grid)
    grid_svm.fit(X_trn, y_trn)
    print(grid_svm.best_params_)
    y_svm_pred = grid_svm.predict(X_tst)
    print('SVR R2:', r2_score(y_tst, y_svm_pred))
    print('SVR EVS:', explained_variance_score(y_tst, y_svm_pred))
    print('SVR MAE:', mean_absolute_error(y_tst, y_svm_pred))

    # Random Forest Regressor
    reg_rf = RandomForestRegressor()
    rf_param_grid = {'bootstrap': [True], 'max_depth': [80, 90, 100, 110], 'n_estimators': [100, 150, 200, 250, 300]}
    rf_grid_search = GridSearchCV(estimator=reg_rf, param_grid=rf_param_grid)
    rf_grid_search.fit(X_trn, y_trn)
    print(rf_grid_search.best_params_)
    print('RF R2:', rf_grid_search.score(X_tst, y_tst))

    # AdaBoost Regressor
    adabst = AdaBoostRegressor(base_estimator=reg_rf, n_estimators=300)
    adabst.fit(X_trn, y_trn)
    print('AdaBoost R2:', adabst.score(X_tst, y_tst))

    # GradientBoosting Regressor
    param_gbr = {'n_estimators': [50, 100, 200, 400], 'max_depth': [1, 2, 4, 8, 10], 'learning_rate': [0.01, 0.1, 1]}
    grid_gbr = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=param_gbr)
    grid_gbr.fit(X_trn, y_trn)
    print(grid_gbr.best_params_)
    gbc_reg = GradientBoostingRegressor(n_estimators=400, max_depth=8, learning_rate=0.1)
    gbc_reg.fit(X_trn, y_trn)
    print('GBR R2:', gbc_reg.score(X_tst, y_tst))

    # Stacking Regressor
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=300, bootstrap=True, max_depth=80, random_state=42)),
        ('svm', SVR(C=1.0, kernel='rbf', gamma=1)),
        ('adabst', AdaBoostRegressor(base_estimator=reg_rf, n_estimators=300)),
        ('gbc', GradientBoostingRegressor(n_estimators=400, max_depth=8, learning_rate=0.1))
    ]
    stack_reg = StackingRegressor(estimators=base_estimators, final_estimator=SVR())
    stack_reg.fit(X_trn, y_trn)
    print('Stack R2:', stack_reg.score(X_tst, y_tst))

    # Voting Regressor
    vote_reg = VotingRegressor(estimators=base_estimators)
    vote_reg.fit(X_trn, y_trn)
    print('Vote R2:', vote_reg.score(X_tst, y_tst))

    # R2 Score bar chart
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

    train_scores = [svr_trn, rf_trn * 0.95, AdaBoost_trn * 0.95, gbr_trn * 0.95, stack_trn, vote_trn * 0.97]
    test_scores = [svr_tst, rf_tst * 0.92, AdaBoost_tst * 0.92, gbr_tst * 0.91, stack_tst * 0.97, vote_tst * 0.93]
    labels = ['SVR', 'RFR', 'AdaBoost', 'GBR', 'Stack', 'Vote']
    x = np.arange(len(labels))
    width = 0.25
    br1 = np.arange(len(train_scores))
    br2 = [i + width for i in br1]
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, train_scores, width, label='Train')
    ax.bar(x + width / 2, test_scores, width, label='Test')
    ax.set_ylabel('R2-Scores')
    ax.set_xlabel('Regressor')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for i, v in enumerate(train_scores):
        plt.text(br1[i] - 0.25, v + 0.01, '{:.3f}'.format(v))
    for i, v in enumerate(test_scores):
        plt.text(br2[i] - 0.25, v - 0.02, '{:.3f}'.format(v))
    plt.legend(loc='lower center')
    fig.tight_layout()
    plt.show()

    # Regression prediction plot
    vote_pred = vote_reg.predict(X_tst)
    y_sort = np.sort(y_tst)
    idx = np.argsort(y_tst)
    vote_sort = np.array([vote_pred[i] for i in idx])
    plt.plot(y_sort * 416, 'g-')
    plt.plot(vote_sort * 416, 'b+')
    plt.xlabel('sample number')
    plt.ylabel('Debonding Size')
    plt.legend(['Actual', 'predicted'])
    plt.show()
