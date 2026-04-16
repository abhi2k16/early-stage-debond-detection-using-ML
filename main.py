"""
main.py — Entry point for the early-stage debond detection pipeline.

Pipeline order:
  Step 1: Train Conv1D Autoencoder          (Dam_cat_1DCNN_AE_TS150ms)
  Step 2: AE + 1D-CNN Classification        (Dam_cat_1DCNN_AE_TS150ms_cls)
  Step 3: AE + 1D-CNN Regression            (Dam_cat_1DCNN_AE_TS150ms_reg)
  Step 4: AE vs PCA Reconstruction Compare  (CAE_PCA_compare)
  Step 5: Simple NN Classification          (Dam_cat_class)

Run individual steps or the full pipeline:
  python main.py --step all
  python main.py --step ae_train
  python main.py --step classify
  python main.py --step regress
  python main.py --step compare
  python main.py --step nn_class
"""

import argparse


def run_ae_train():
    print("\n=== Step 1: Conv1D Autoencoder Training ===")
    import numpy as np
    import torch
    import copy
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from torch.utils.data import DataLoader
    from Dam_cat_1DCNN_AE_TS150ms import Conv1D_AE, train, test_model

    data = loadmat('X_FEATURE_snr55_60_65.mat')
    X_Feature = data.get('X_FEATURE_snr55_60_65')
    idx = np.arange(len(X_Feature))
    np.random.shuffle(idx)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    x_feture_tensor = torch.from_numpy(X_Feature)
    x_train = torch.zeros_like(x_feture_tensor)
    for i in idx:
        x_train[i] = x_feture_tensor[i]
    x_train_tensor = x_train.reshape(len(idx), 1, 41).float().to(device)

    AE_model = Conv1D_AE().to(device)
    intial_weight = copy.deepcopy(AE_model.fc_encoder[6].weight)
    plt.imshow(intial_weight.detach().numpy())
    plt.title('Initial Encoder Weights')
    plt.show()

    train(AE_model, x_train_tensor, num_epochs=15000, batch_size=120, lr=0.005)

    train_weight = copy.deepcopy(AE_model.fc_encoder[6].weight)
    plt.imshow(train_weight.detach().numpy())
    plt.title('Trained Encoder Weights')
    plt.show()

    update_weight = train_weight - intial_weight
    plt.imshow(update_weight.detach().numpy())
    plt.title('Weight Update')
    plt.show()

    n = 1
    x = x_train_tensor[n].reshape(1, 1, 41)
    test_model(AE_model, x_train_tensor, x, n)
    print("Step 1 complete.")


def run_classify():
    print("\n=== Step 2: AE + 1D-CNN Classification ===")
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split
    from ConvAE_BNL_16 import Conv1D_AE_BNL_16
    from Dam_cat_1DCNN_AE_TS150ms_cls import nn_model, get_activation, train_model, evaluate
    from roc_auc_utils import ROC_Curve_avg, ROC_Curve_all, precision_recall_all, precision_recall_avg
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.svm import SVC
    from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                   GradientBoostingClassifier, StackingClassifier, VotingClassifier)
    from sklearn.model_selection import GridSearchCV
    from sklearn.decomposition import PCA
    from mlxtend.plotting import plot_confusion_matrix

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

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
    train_model(Cls_model, X_train, y_train, epoch=2000, lr=0.05, batch_size=151)

    class_name = ['1', '2', '3', '4']
    for X_data, y_data, title in [(X_train, y_train, 'Train'), (X_test, y_test, 'Test')]:
        y_h = Cls_model(X_data)
        y_pred_arr = y_h.max(1, keepdim=True)[1].squeeze().numpy()
        y_act_arr = y_data.max(1, keepdim=True)[1].squeeze().numpy()
        con_mat = confusion_matrix(y_act_arr, y_pred_arr)
        fig, ax = plot_confusion_matrix(conf_mat=con_mat, colorbar=True, show_absolute=True,
                                        show_normed=True, class_names=class_name, figsize=[8, 8])
        plt.title(title)
        plt.show()

    y_h = Cls_model(X_test)
    y_score = y_h.detach().numpy()
    y_true = y_test.numpy()
    y_pred_arr = y_h.max(1, keepdim=True)[1].squeeze().numpy()
    y_act_arr = y_test.max(1, keepdim=True)[1].squeeze().numpy()
    print(classification_report(y_act_arr, y_pred_arr, labels=(0, 1, 2, 3), target_names=class_name))

    pca = PCA(n_components=4)
    X = pca.fit_transform(X_AE_feature.numpy())
    y = T_class.numpy()
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.15, random_state=42)
    encoder = OneHotEncoder()
    y_true_ohe = encoder.fit_transform(y_tst.reshape(-1, 1)).toarray()
    class_name_zones = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4']

    param_grid = {'C': [1, 10, 100, 1000, 10000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    grid_svm = GridSearchCV(SVC(random_state=42, probability=True), param_grid)
    grid_svm.fit(X_trn, y_trn)
    y_svm_pred = grid_svm.predict(X_tst)
    y_svm_prob = grid_svm.predict_proba(X_tst)

    cl_rf = RandomForestClassifier(random_state=42)
    rf_grid_search = GridSearchCV(cl_rf, {'bootstrap': [True], 'max_depth': [80, 90, 100, 110],
                                           'n_estimators': [100, 150, 200, 250, 300]}, cv=3)
    rf_grid_search.fit(X_trn, y_trn)
    rf_best_grid = rf_grid_search.best_estimator_
    y_rf_pred = rf_best_grid.predict(X_tst)
    y_rf_prob = rf_best_grid.predict_proba(X_tst)

    adabst = AdaBoostClassifier(base_estimator=cl_rf, n_estimators=300, random_state=42)
    adabst.fit(X_trn, y_trn)
    y_adabst_prob = adabst.predict_proba(X_tst)
    y_adabst_pred = adabst.predict(X_tst)

    gbc_clf = GradientBoostingClassifier(n_estimators=50, max_depth=8, learning_rate=0.1)
    gbc_clf.fit(X_trn, y_trn)
    y_gbc_prob = gbc_clf.predict_proba(X_tst)
    y_gbc_pred = gbc_clf.predict(X_tst)

    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, bootstrap=True, max_depth=80, random_state=42)),
        ('svm', SVC(C=1.0, kernel='rbf', gamma=1, probability=True)),
        ('adabst', AdaBoostClassifier(base_estimator=cl_rf, n_estimators=300, random_state=42)),
        ('gbc', GradientBoostingClassifier(n_estimators=50, max_depth=8, learning_rate=0.1))
    ]
    stack_clf = StackingClassifier(estimators=base_estimators, stack_method='predict_proba',
                                   final_estimator=SVC(probability=True))
    stack_clf.fit(X_trn, y_trn)
    y_stc_prob = stack_clf.predict_proba(X_tst)

    vote_clf = VotingClassifier(estimators=base_estimators, voting='soft')
    vote_clf.fit(X_trn, y_trn)
    y_vote_prob = vote_clf.predict_proba(X_tst)

    clf_names = ['SVM', 'rf', 'adaboost', 'GBC', 'stack', 'vote']
    y_probs = [y_svm_prob, y_rf_prob, y_adabst_prob, y_gbc_prob, y_stc_prob, y_vote_prob]
    colors = ['red', 'green', 'blue', 'black', 'orange', 'maroon']
    for i in range(6):
        ROC_Curve_avg(y_true_ohe, y_probs[i], name=clf_names[i], color=colors[i])
    plt.show()
    for i in range(6):
        precision_recall_avg(y_true_ohe, y_probs[i], name=clf_names[i], color=colors[i])
    plt.show()
    print("Step 2 complete.")


def run_regress():
    print("\n=== Step 3: AE + 1D-CNN Regression ===")
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from Dam_cat_1DCNN_AE_TS150ms_reg import Conv1D_AE, nn_model_reg, get_activation, train_model
    from sklearn.svm import SVR
    from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                                   GradientBoostingRegressor, StackingRegressor, VotingRegressor)
    from sklearn.model_selection import GridSearchCV

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    data = loadmat('X_FEATURE_snr35_40.mat')
    X_Feature = data.get('X_FEATURE_snr35_40')

    AE_model_trained = Conv1D_AE()
    AE_model_trained.load_state_dict(torch.load('AE_model_snr35_40.pth'))
    AE_model_trained.eval()

    activation = {}
    X_AE_input = torch.from_numpy(X_Feature).view(2688, 1, 41).float().to(device)
    AE_feature = []
    for i in X_AE_input:
        AE_model_trained.fc_encoder.register_forward_hook(get_activation(activation, 'fc_encoder'))
        x = i.view(1, 1, 41)
        AE_model_trained(x)
        AE_feature.append(activation['fc_encoder'][0].view(16))

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
    y_test_t = torch.from_numpy(y_test).float().to(device)

    Cls_model = nn_model_reg().to(device)
    train_model(Cls_model, X_train, y_train, epoch=1000, lr=0.1, batch_size=1344)

    X = X_Feature
    y = X_Reg.ravel()
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.09, random_state=42)

    reg_rf = RandomForestRegressor()
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=300, bootstrap=True, max_depth=80, random_state=42)),
        ('svm', SVR(C=1.0, kernel='rbf', gamma=1)),
        ('adabst', AdaBoostRegressor(base_estimator=reg_rf, n_estimators=300)),
        ('gbc', GradientBoostingRegressor(n_estimators=400, max_depth=8, learning_rate=0.1))
    ]
    vote_reg = VotingRegressor(estimators=base_estimators)
    vote_reg.fit(X_trn, y_trn)
    print('Vote R2:', vote_reg.score(X_tst, y_tst))

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
    print("Step 3 complete.")


def run_compare():
    print("\n=== Step 4: AE vs PCA Reconstruction Comparison ===")
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from sklearn.decomposition import PCA
    from sklearn.utils import shuffle
    from CAE_PCA_compare import cal_mse_pca, cal_mse_pca_2
    from ConvAE_BNL_4 import Conv1D_AE_BNL_4
    from ConvAE_BNL_8 import Conv1D_AE_BNL_8
    from ConvAE_BNL_16 import Conv1D_AE_BNL_16

    data = loadmat('X_FEATURE_snr55_60_65.mat')
    X_Feature = data.get('X_FEATURE_snr55_60_65')
    x_tensor = torch.from_numpy(X_Feature).view(len(X_Feature), 1, 41).float()

    AE_model_trained_1 = Conv1D_AE_BNL_4()
    AE_model_trained_2 = Conv1D_AE_BNL_8()
    AE_model_trained_3 = Conv1D_AE_BNL_16()
    AE_model_trained_1.load_state_dict(torch.load('AE_CF_4_snr55_60_65-2.pth'))
    AE_model_trained_2.load_state_dict(torch.load('AE_CF_8_snr55_60_65-3.pth'))
    AE_model_trained_3.load_state_dict(torch.load('AE_CF_16_snr55_60_65-2.pth'))
    AE_model_trained_1.eval()
    AE_model_trained_2.eval()
    AE_model_trained_3.eval()

    X_Feature = x_tensor.numpy().reshape(3600, 41)
    n_com1, n_com2 = 4, 15
    pca = PCA(n_components=n_com1, svd_solver='randomized')
    pca_2 = PCA(n_components=n_com2, svd_solver='randomized')
    X_train_pca = pca.fit_transform(X_Feature)
    X_train_pca_2 = pca_2.fit_transform(X_Feature)

    X_Train_Pca = np.concatenate((X_train_pca, X_train_pca_2), axis=1)
    X_Feature, X_Train_Pca = shuffle(X_Feature, X_Train_Pca)
    X_train_pca = X_Train_Pca[:, 0:n_com1]
    X_train_pca_2 = X_Train_Pca[:, n_com1:n_com1 + n_com2]
    x_train_tensor = torch.from_numpy(X_Feature).reshape(len(X_Feature), 1, 41).float()

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
    print("Step 4 complete.")


def run_nn_class():
    print("\n=== Step 5: Simple NN Classification ===")
    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split
    from torch.autograd import Variable
    from Dam_cat_class import nn_model, train, saveModel

    data = loadmat('X_feature.mat')
    target = loadmat('T_reg.mat')
    X_Feature = data.get('X_Feature')
    target = target.get('T_reg').reshape(177)
    target_1 = np.array(target, dtype='float') / np.max(target)

    T_class = torch.arange(0, 177)
    for i in range(177):
        if i <= 75:
            T_class[i] = 1
        elif i <= 105:
            T_class[i] = 2
        elif i <= 150:
            T_class[i] = 3
        else:
            T_class[i] = 4
    T_class = F.one_hot(T_class % 4)

    X_train, X_test, y_train, y_test = train_test_split(X_Feature, T_class, test_size=0.1, random_state=42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    from torch.optim import Adam
    import torch.nn as nn
    model = nn_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    training_loss = train(model, optimizer, loss_fn, X_train_tensor, y_train, epochs=5, batch_size=159)
    saveModel(model)

    plt.plot(range(len(training_loss)), training_loss, 'g-', label='Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.show()
    print("Step 5 complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debond Detection Pipeline')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'ae_train', 'classify', 'regress', 'compare', 'nn_class'],
                        help='Pipeline step to run')
    args = parser.parse_args()

    steps = {
        'ae_train': run_ae_train,
        'classify': run_classify,
        'regress':  run_regress,
        'compare':  run_compare,
        'nn_class': run_nn_class,
    }

    if args.step == 'all':
        for name, fn in steps.items():
            fn()
    else:
        steps[args.step]()
