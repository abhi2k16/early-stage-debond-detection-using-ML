from ConvAE_BNL_16 import Conv1D_AE_BNL_16
from roc_auc_utils import ROC_Curve_avg, ROC_Curve_all, precision_recall_all, precision_recall_avg
import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, StackingClassifier, VotingClassifier)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_confusion_matrix


class nn_model(nn.Module):
    def __init__(self):
        super(nn_model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, stride=3)
        self.BN1 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16 * 5, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, x):
        x = x.view(-1, 1, 16)
        x = self.BN1(self.conv1(x))
        x = x.view(-1, 16 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def get_activation(activation, name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def train_model(Cls_model, X_train, y_train, epoch, lr, batch_size):
    loss_fn = nn.CrossEntropyLoss()
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


def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    error = abs(pred - y_test)
    mape = 100 * np.mean(error / y_test)
    acc = 100 - mape
    print('Model_performance')
    print('Average error : {:0.4f} degree.'.format(np.mean(error)))
    print('Accuracy = {:0.2f}%'.format(acc))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data = loadmat('X_FEATURE_snr55_60_65.mat')
    X_feature = data.get('X_FEATURE_snr55_60_65')
    X_Data = X_feature[0:1200, :]

    file = 'AE_CF_16_snr55_60_65-2.pth'
    AE_model_trained = Conv1D_AE_BNL_16()
    AE_model_trained.load_state_dict(torch.load(file))
    AE_model_trained.eval()

    activation = {}
    X_AE_input = torch.from_numpy(X_Data).view(1200, 1, 41).float().to(device)
    AE_feature = []
    for i in X_AE_input:
        AE_model_trained.fc_encoder.register_forward_hook(get_activation(activation, 'fc_encoder'))
        x = i.view(1, 1, 41)
        AE_model_trained(x)
        AE_feature.append(activation['fc_encoder'][0].view(16))
        plt.plot(activation['fc_encoder'][0].numpy())
    plt.show()

    X_AE_feature = torch.zeros(1200, 16)
    for i in range(len(AE_feature)):
        X_AE_feature[i] = AE_feature[i]

    T_class = torch.arange(0, 1200)
    for i in range(1200):
        if i <= 400:
            T_class[i] = 0
        elif i <= 736:
            T_class[i] = 1
        elif i <= 1056:
            T_class[i] = 2
        else:
            T_class[i] = 3
    Tar_class = F.one_hot(T_class, num_classes=4)

    X_train, X_test, y_train, y_test = train_test_split(X_AE_feature, Tar_class, test_size=0.1, random_state=42)
    X_train = X_train.float().to(device)
    X_test = X_test.float().to(device)
    y_train = y_train.float().to(device)

    Cls_model = nn_model().to(device)
    lr = 0.05
    batch_size = 151
    max_epoch = 2000
    train_model(Cls_model, X_train, y_train, epoch=max_epoch, lr=lr, batch_size=batch_size)

    class_name = ['1', '2', '3', '4']
    for X_data, y_data, title in [(X_train, y_train, 'Train'), (X_test, y_test, 'Test')]:
        y_h = Cls_model(X_data)
        y_pred = y_h.max(1, keepdim=True)[1]
        y_act = y_data.max(1, keepdim=True)[1]
        y_pred_arr = y_pred.squeeze().numpy()
        y_act_arr = y_act.squeeze().numpy()
        con_mat = confusion_matrix(y_act_arr, y_pred_arr)
        fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=True,
                                        show_absolute=True, show_normed=True,
                                        class_names=class_name, figsize=[8, 8])
        plt.title(title)
        plt.show()

    y_h = Cls_model(X_test)
    y_score = y_h.detach().numpy()
    y_true = y_test.numpy()
    y_pred_arr = y_h.max(1, keepdim=True)[1].squeeze().numpy()
    y_act_arr = y_test.max(1, keepdim=True)[1].squeeze().numpy()
    cls_rpt = classification_report(y_act_arr, y_pred_arr, labels=(0, 1, 2, 3), target_names=class_name)
    print(cls_rpt)

    # SVM
    pca = PCA(n_components=4)
    X = pca.fit_transform(X_AE_feature.numpy())
    y = T_class.numpy()
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.15, random_state=42)

    param_grid = {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    grid_svm = GridSearchCV(SVC(random_state=42, probability=True), param_grid)
    grid_svm.fit(X_trn, y_trn)
    print(grid_svm.best_params_)
    y_svm_pred = grid_svm.predict(X_tst)
    y_svm_prob = grid_svm.predict_proba(X_tst)
    print('SVM Train:', accuracy_score(y_trn, grid_svm.predict(X_trn)))
    print('SVM Test:', accuracy_score(y_tst, y_svm_pred))

    encoder = OneHotEncoder()
    y_true_ohe = encoder.fit_transform(y_tst.reshape(-1, 1)).toarray()
    class_name_zones = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']

    con_mat = confusion_matrix(y_tst, y_svm_pred)
    fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=False, show_absolute=True,
                                    show_normed=True, class_names=class_name_zones, figsize=[4, 4])
    plt.title('SVM', fontweight='bold', fontsize=10)
    plt.show()
    ROC_Curve_all(y_true_ohe, y_svm_prob, clf_name='SVM')
    precision_recall_all(y_true_ohe, y_svm_prob, clf_name='svm')

    # Random Forest
    cl_rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {'bootstrap': [True], 'max_depth': [80, 90, 100, 110], 'n_estimators': [100, 150, 200, 250, 300]}
    rf_grid_search = GridSearchCV(estimator=cl_rf, param_grid=rf_param_grid, cv=3)
    rf_grid_search.fit(X_trn, y_trn)
    print(rf_grid_search.best_params_)
    rf_best_grid = rf_grid_search.best_estimator_
    evaluate(rf_best_grid, X_tst, y_tst)
    y_rf_pred = rf_best_grid.predict(X_tst)
    y_rf_prob = rf_best_grid.predict_proba(X_tst)
    con_mat = confusion_matrix(y_tst, y_rf_pred)
    fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=False, show_absolute=True,
                                    show_normed=True, class_names=class_name_zones, figsize=[4, 4])
    plt.title('RF', fontweight='bold', fontsize=10)
    plt.show()
    ROC_Curve_all(y_true_ohe, y_rf_prob, clf_name='rf')
    precision_recall_all(y_true_ohe, y_rf_prob, clf_name='rf')

    # AdaBoost
    adabst = AdaBoostClassifier(base_estimator=cl_rf, n_estimators=300, random_state=42)
    adabst.fit(X_trn, y_trn)
    y_adabst_prob = adabst.predict_proba(X_tst)
    y_adabst_pred = adabst.predict(X_tst)
    ROC_Curve_all(y_true_ohe, y_adabst_prob, clf_name='adaboost')
    precision_recall_all(y_true_ohe, y_adabst_prob, clf_name='adaboost')
    con_mat = confusion_matrix(y_tst, y_adabst_pred)
    fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=False, show_absolute=True,
                                    show_normed=True, class_names=class_name_zones, figsize=[4, 4])
    plt.title('AdaBoost', fontweight='bold', fontsize=10)
    plt.show()

    # GradientBoosting
    param_gbc = {'n_estimators': [50, 100, 200, 400], 'max_depth': [1, 2, 4, 8, 16],
                 'learning_rate': [0.01, 0.1, 1, 10, 100]}
    grid_gbc = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_gbc, cv=5)
    grid_gbc.fit(X_trn, y_trn)
    print(grid_gbc.best_params_)
    gbc_clf = GradientBoostingClassifier(n_estimators=50, max_depth=8, learning_rate=0.1)
    gbc_clf.fit(X_trn, y_trn)
    y_gbc_prob = gbc_clf.predict_proba(X_tst)
    y_gbc_pred = gbc_clf.predict(X_tst)
    ROC_Curve_all(y_true_ohe, y_gbc_prob, clf_name='grnboost')
    precision_recall_all(y_true_ohe, y_gbc_prob, clf_name='grnboost')
    con_mat = confusion_matrix(y_tst, y_gbc_pred)
    fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=False, show_absolute=True,
                                    show_normed=True, class_names=class_name_zones, figsize=[4, 4])
    plt.title('GBC', fontweight='bold', fontsize=10)
    plt.show()

    # Stacking
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, bootstrap=True, max_depth=80, random_state=42)),
        ('svm', SVC(C=1.0, kernel='rbf', gamma=1, probability=True)),
        ('adabst', AdaBoostClassifier(base_estimator=cl_rf, n_estimators=300, random_state=42)),
        ('gbc', GradientBoostingClassifier(n_estimators=50, max_depth=8, learning_rate=0.1))
    ]
    stack_clf = StackingClassifier(estimators=base_estimators, stack_method='predict_proba',
                                   final_estimator=SVC(probability=True))
    stack_clf.fit(X_trn, y_trn)
    y_pred_stack = stack_clf.predict(X_tst)
    y_stc_prob = stack_clf.predict_proba(X_tst)
    con_mat = confusion_matrix(y_tst, y_pred_stack)
    fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=False, show_absolute=True,
                                    show_normed=True, class_names=class_name_zones, figsize=[4, 4])
    plt.title('stack', fontweight='bold', fontsize=10)
    plt.show()
    ROC_Curve_all(y_true_ohe, y_stc_prob, clf_name='stack')
    precision_recall_all(y_true_ohe, y_stc_prob, clf_name='stack')

    # Voting
    vote_clf = VotingClassifier(estimators=base_estimators, voting='soft')
    vote_clf.fit(X_trn, y_trn)
    y_vote_pred = vote_clf.predict(X_tst)
    y_vote_prob = vote_clf.predict_proba(X_tst)
    con_mat = confusion_matrix(y_tst, y_vote_pred)
    fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=False, show_absolute=True,
                                    show_normed=True, class_names=class_name_zones, figsize=[4, 4])
    plt.title('vote', fontweight='bold', fontsize=10)
    plt.show()
    ROC_Curve_all(y_true_ohe, y_vote_prob, clf_name='vote')
    precision_recall_all(y_true_ohe, y_vote_prob, clf_name='vote')

    # Comparison ROC & Precision-Recall
    clf_names = ['SVM', 'rf', 'adaboost', 'GBC', 'stack', 'vote']
    y_probs = [y_svm_prob, y_rf_prob, y_adabst_prob, y_gbc_prob, y_stc_prob, y_vote_prob]
    colors = ['red', 'green', 'blue', 'black', 'orange', 'maroon']
    for i in range(6):
        ROC_Curve_avg(y_true_ohe, y_probs[i], name=clf_names[i], color=colors[i])
    plt.show()
    for i in range(6):
        precision_recall_avg(y_true_ohe, y_probs[i], name=clf_names[i], color=colors[i])
    plt.show()
