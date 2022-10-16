from ConvAE_BNL_16 import Conv1D_AE_BNL_16
import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable 
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
# check device
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data = loadmat('X_FEATURE_snr55_60_65.mat')
X_feature = data.get('X_FEATURE_snr55_60_65')
X_Data = X_feature[0:1200,:]
# Load model structure and trained model weight

###############################################################################
########################### DataSset ##########################################
###############################################################################
AE_trained_model = Conv1D_AE_BNL_16()

file = 'AE_CF_16_snr55_60_65-2.pth'
AE_model_trained = Conv1D_AE_BNL_16()
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
X_AE_input = torch.from_numpy(X_Data).view(1200,1,41).float().to(device)

AE_feature = []
for i in X_AE_input:
    AE_model_trained.fc_encoder.register_forward_hook(get_activation('fc_encoder'))
    x = i.view(1,1,41)
    X = AE_model_trained(x)
    AE_feature.append(activation['fc_encoder'][0].view(16))
    plt.plot(activation['fc_encoder'][0].numpy())
plt.show()

# take out tensor from X_AE_out list
X_AE_feature = torch.zeros(1200,16)
for i in range(len(AE_feature)):
    X_AE_feature[i] = AE_feature[i]


# Targer value for class
T_class = torch.arange(0,1200)
for i in range(1200):
    if i <= 400:
        T_class[i] = 0
    elif i > 400 and i <= 736:
        T_class[i] = 1
        
    elif i > 736 and i <= 1056:
        T_class[i] = 2
    elif i > 1056:
        T_class[i] = 3
# Change the class value into one_hot encoder 
import torch.nn.functional as F
Tar_class = F.one_hot(T_class, num_classes=4)
#Tar_class = torch.cat((Tar_class, Tar_class), 0)

# split the train and test, input and label data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_AE_feature, Tar_class, test_size=0.1, random_state=42)
# convert ndarray into torch tensor
X_train = X_train.float().to(device)
X_test = X_test.float().to(device)
y_train = y_train.float().to(device)
###############################################################################
############################### 1D-CNN ########################################
###############################################################################
# Classifier model
class nn_model(nn.Module):
    def __init__(self):
        super(nn_model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, stride=3)
        self.BN1 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16*5,40)
        self.fc2 = nn.Linear(40,20)
        self.fc3 = nn.Linear(20, 4)

    def forward(self, x):
        x = x.view(-1, 1, 16)
        x = self.BN1(self.conv1(x))
        x = x.view(-1, 16*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

Cls_model = nn_model().to(device)

# Defining training routine
#loss_fn = nn.MSELoss()
#loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.L1Loss()
#optim_cls = torch.optim.SGD(Cls_model.parameters(), lr = lr, momentum= 0.9)
#optim_cls = optim.Adagrad(Cls_model.parameters(), lr=lr)
#optim_cls = optim.Adam(Cls_model.parameters(), lr=lr, weight_decay=0.001)

# training
def train_model(epoch, lr, batch_size, model):
    loss_fn = nn.CrossEntropyLoss()
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
lr = 0.05 
batch_size = 151
max_epoch = 2000
outputs = train_model(epoch = max_epoch,lr = lr,batch_size=batch_size, model=Cls_model)
# Test moel with train data
X_data=X_train
y_h=Cls_model(X_data)
y_pred=y_h.max(1,keepdim=True)[1]
y_act=y_train.max(1,keepdim=True)[1]
y_pred_arr=y_pred.squeeze().numpy()
y_act_arr=y_act.squeeze().numpy()
# CONFUSION MATRIX
con_mat=confusion_matrix(y_act_arr,y_pred_arr)
#print(con_mat)
class_name=['1','2','3','4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=True,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[8, 8])
plt.show()
# Test
X_data=X_test
y_h=Cls_model(X_data)
y_pred=y_h.max(1,keepdim=True)[1]
y_act=y_test.max(1,keepdim=True)[1]
y_pred_arr=y_pred.squeeze().numpy()
y_act_arr=y_act.squeeze().numpy()
# CONFUSION MATRIX
con_mat=confusion_matrix(y_act_arr,y_pred_arr)
#print(con_mat)
class_name=['1','2','3','4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=True,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[8, 8])
plt.show()
###############################################################################
############################# ROC & AUC #######################################
###############################################################################
from sklearn.metrics import roc_curve, auc, classification_report
from itertools import cycle
cls_rpt = classification_report(y_act_arr,y_pred_arr,labels=(0,1,2,3), target_names=class_name)
#y_score = f1_score(y_act_arr,y_pred_arr, average= None)
y_score = y_h.detach().numpy()  #Get the class prob 
y_true = y_test.numpy()
n_classes = 4
################### Function For ROC Curve estimetion  ########################
def ROC_Curve_avg(y_true, y_prob_pred, name, color):
    clf_name = name
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_prob_pred[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
    # compute micro-avarage ROC and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_prob_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    # Plot ROC for all class
    # Average all it and compute AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curve at this point
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    # Plot all roc
    area = roc_auc['micro']
    lw = 2
    plt.plot(
        fpr['micro'],
        tpr['micro'],
        #label = 'ROC (area ={0:02f}) '.format( roc_auc['micro']),
        label = f'{clf_name} ROC (area = {"%0.04f"%area})',
        color = color,
        linestyle = '-',
        linewidth = 2,
        )
    plt.plot([0,1],[0,1],'k--', lw= lw)
    plt.xlim(-0.01,1)
    plt.ylim((0,1.05))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
# ROC Curve for all class and average of all
def ROC_Curve_all(y_true, y_prob_pred, clf_name):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_prob_pred[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
    # compute micro-avarage ROC and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_prob_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    # Plot ROC for all class
    # Average all it and compute AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curve at this point
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    # Plot all roc
    lw = 2
    plt.figure()
    plt.plot(
        fpr['micro'],
        tpr['micro'],
        label = 'micro-average ROC (area ={0:04f})'.format(roc_auc['micro']),
        color = 'deeppink',
        linestyle = ':',
        linewidth = 2,
        )
    # ROC for individual class
    color = cycle(['aqua','darkorange','cornflowerblue','maroon'])
    for i, color in zip(range(n_classes),color):
        plt.plot(
            fpr[i],
            tpr[i],
            color = color,
            lw= lw,
            label = 'ROC Zone {0} (area = {1:0.4f})'.format(i+1, roc_auc[i]),
            )
    plt.plot([0,1],[0,1],'k--', lw= lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
    plt.title(clf_name)
    plt.show()
###############################################################################
######################### Pecision recall curve ###############################
###############################################################################
from sklearn.metrics import precision_recall_curve, average_precision_score
################## plotting precision recall all classes ######################
def precision_recall_all(y_true, y_prob, clf_name):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true[:,i], 
            y_prob[:,i])
        average_precision[i] = average_precision_score(
            y_true[:,i], 
            y_prob[:,i])
    # Compute micro avg ROC curve and ROC area
    precision['micro'], recall['micro'],_=precision_recall_curve(
        y_true.ravel(), 
        y_prob.ravel())
    average_precision['micro'] = average_precision_score(
        y_true, 
        y_prob,
        average='micro')
    # Plot average precision curve
    plt.plot(
        recall['micro'],
        precision['micro'],
        label = 'Average (area = {0:0.3f})'.format(average_precision['micro'])
        )
    # Plot precision recall for each class
    for i in range(n_classes):
        plt.plot(
            recall[i], 
            precision[i], lw=2,
            label = 'Zone {0} (area {1:0.4f})'.format(i+1, average_precision[i]))
    # Plot avg precision
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc='best')
    plt.title(clf_name)
    plt.show()
################## Function for plotting precision recall avg #################
def precision_recall_avg(y_true, y_prob, name, color):
    clf_name = name
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true[:,i], 
            y_prob[:,i])
        average_precision[i] = average_precision_score(
            y_true[:,i], 
            y_prob[:,i])
    # Compute micro avg ROC curve and ROC area
    precision['micro'], recall['micro'],_=precision_recall_curve(
        y_true.ravel(), 
        y_prob.ravel())
    average_precision['micro'] = average_precision_score(
        y_true, 
        y_prob,
        average='micro')
    # Plot average precision curve
    area = average_precision['micro']
    plt.plot(
        recall['micro'],
        precision['micro'],
        label = f'{clf_name} (area = {"%0.04f"%area})'
        )
    # Plot avg precision
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc='best')

###############################################################################
############################### SVM ###########################################
###############################################################################
from sklearn.svm import SVC
rng = np.random.RandomState(0)
X = X_AE_feature.numpy()
#X = X_Feature
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X = pca.fit_transform(X)
#y =torch.cat((T_class, T_class), 0).numpy() 
y = T_class.numpy()
#y = Tar_class.numpy() 
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.15, random_state=42)

clf_svm = SVC(random_state=42,probability=True)
from sklearn.model_selection import GridSearchCV
#Create a svm Classifier and hyper parameter tuning 
# defining parameter range
param_grid = {'C': [ 1, 10, 100, 1000,10000], 
              'gamma': [1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf']} 
  
grid_svm = GridSearchCV(clf_svm, param_grid)
# fitting the model for grid search
grid_svm.fit(X_trn, y_trn)
print(grid_svm.best_params_)
# {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
#clf.fit(X_trn, y_trn)
y_svm_pred = grid_svm.predict(X_tst)
y_svm_prob = grid_svm.predict_proba(X_tst)
y = y_tst
from sklearn.metrics import accuracy_score
svm_trn_score = accuracy_score(y_trn, grid_svm.predict(X_trn))
svm_tst_score = accuracy_score(y_tst, grid_svm.predict(X_tst)) 
con_mat=confusion_matrix(y_tst,y_svm_pred)
#print(con_mat)
class_name=['Zone 1','Zone 2','Zone 3','Zone 4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[4, 4])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('SVM',fontweight = 'bold',fontsize = 10)
plt.show()
################## ROC AUC #####################################
############# Compute POC curve and ROC area for each class ###################
# convert class label in one hot encoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y = y_tst.reshape(len(y_tst),1)
y_true = encoder.fit_transform(y).toarray()
ROC_Curve_all(y_true, y_prob_pred=y_svm_prob, clf_name = 'SVM')
precision_recall_all(y_true, y_prob=y_svm_prob, clf_name='svm')
###############################################################################
####################### RandomForestClass #####################################
###############################################################################
from sklearn.ensemble import RandomForestClassifier
# Create a base model
cl_rf = RandomForestClassifier(random_state= 42)
print(cl_rf.get_params())
# Hyperparameter tuning using Gridsearch
# Create parameter grid
rf_param_grid = {
    'bootstrap' : [True],
    'max_depth' : [80, 90, 100, 110],
    'n_estimators' : [100, 150, 200, 250, 300]
    }
# Intant gread search model
rf_grid_search = GridSearchCV(estimator=cl_rf, param_grid=rf_param_grid, cv = 3)
# Fit grid search to data
rf_grid_search.fit(X_trn, y_trn)
print(rf_grid_search.best_params_)
#{'bootstrap': True, 'max_depth': 80, 'n_estimators': 300}
rf_best_grid = rf_grid_search.best_estimator_
#def evalutor function
def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    error = abs(pred - y_test)
    mape = 100*np.mean(error/y_test)
    acc = 100-mape
    print('Model_performance')
    print('Average error : {:0.4f} degree.'.format(np.mean(error)))
    print('Accuracy = {:0.2f}%'.format(acc))

grid_acc = evaluate(model=rf_best_grid, x_test=X_tst, y_test=y_tst)
y_rf_pred = rf_best_grid.predict(X_tst)
con_mat=confusion_matrix(y,y_rf_pred)
#print(con_mat)
class_name=['Zone 1','Zone 2','Zone 3','Zone 4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[4, 4])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('RF',fontweight = 'bold',fontsize = 10)
plt.show()
# PLot ROC 
# Calculte the class prob
y_rf_prob = rf_best_grid.predict_proba(X_tst)
ROC_Curve_all(y_true, y_prob_pred=y_rf_prob, clf_name='rf')
precision_recall_all(y_true, y_prob=y_rf_prob, clf_name= 'rf') 
###############################################################################
##################### AdaBoostClassfier  ######################################
###############################################################################
from sklearn.ensemble import AdaBoostClassifier
adabst = AdaBoostClassifier(
    base_estimator=(cl_rf),
    n_estimators=300, 
    random_state=42)
adabst.fit(X_trn, y_trn)
y_adabst_prob = adabst.predict_proba(X_tst)

ROC_Curve_all(y_true, y_prob_pred=y_adabst_prob, clf_name='adaboost')
precision_recall_all(y_true, y_prob=y_adabst_prob, clf_name = 'adaboost')
y_adabst_pred = adabst.predict(X_tst)
con_mat=confusion_matrix(y,y_adabst_pred)
#print(con_mat)
class_name=['Zone 1','Zone 2','Zone 3','Zone 4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[4, 4])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('AdaBoost',fontweight = 'bold',fontsize = 10)
plt.show()
###############################################################################
##################### GrandBoostClassifier ####################################
###############################################################################
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
param_gbc = {
    'n_estimators' : [50, 100, 200, 400],
    'max_depth' : [1,2,4,8,16],
    'learning_rate' : [0.01, 0.1, 1, 10, 100]
    }
grid_gbc = GridSearchCV(estimator=gbc, param_grid = param_gbc, cv = 5)
grid_gbc.fit(X_trn, y_trn)
print(grid_gbc.best_params_)
# {'learning_rate': 1, 'max_depth': 8, 'n_estimators': 50}
gbc_best_param = grid_gbc.best_params_ 
# 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 200
gbc_clf = GradientBoostingClassifier(
    n_estimators=50, 
    max_depth=8, 
    learning_rate= 0.1)
gbc_clf.fit(X_trn, y_trn)
y_gbc_prob = gbc_clf.predict_proba(X_tst)
ROC_Curve_all(y_true, y_prob_pred=y_gbc_prob, clf_name='grnboost')
precision_recall_all(y_true, y_prob=y_gbc_prob, clf_name = 'grnboost')

y_gbc_pred = gbc_clf.predict(X_tst)
con_mat=confusion_matrix(y,y_gbc_pred)
#print(con_mat)
class_name=['Zone 1','Zone 2','Zone 3','Zone 4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[4, 4])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('GBC',fontweight = 'bold',fontsize = 10)
plt.show()
###############################################################################
##################### Stacked Classifier ######################################
###############################################################################
from sklearn.ensemble import StackingClassifier
base_estimator = [
    ('rf', RandomForestClassifier(n_estimators=300, bootstrap=True, max_depth= 80, random_state=42)),
    ('svm', SVC(C=1.0,kernel='rbf',gamma=1, probability=True)),
    ('adabst', AdaBoostClassifier(
        base_estimator=(cl_rf),
        n_estimators=300, random_state=42)),
    ('gbc', GradientBoostingClassifier(
        n_estimators=50,
        max_depth=8,
        learning_rate=0.1
        ))
    ]
stack_clf = StackingClassifier(
    estimators=base_estimator, 
    stack_method = 'predict_proba',                          
    final_estimator= SVC(probability=True)
    )
stack_clf.fit(X_trn, y_trn)
y_pred_stack = stack_clf.predict(X_tst)
# Plot confusion matrix
con_mat=confusion_matrix(y_tst,y_pred_stack)
#print(con_mat)
class_name=['Zone 1','Zone 2','Zone 3','Zone 4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[4, 4])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('stack',fontweight = 'bold',fontsize = 10)
plt.show()
####################### ROC for stacked classifier ############################
y_stc_prob = stack_clf.predict_proba(X_tst)
ROC_Curve_all(y_true, y_prob_pred=y_stc_prob, clf_name='stack')
precision_recall_all(y_true, y_prob=y_stc_prob, clf_name='stack')
###############################################################################
########################### VotingClassifier ##################################
###############################################################################
from sklearn.ensemble import VotingClassifier
base_estimator = [
    ('rf', RandomForestClassifier(
        n_estimators=300, 
        bootstrap=True, 
        random_state=42)),
    ('svm', SVC(C=1.0,kernel='rbf',gamma=1, probability=True)),
    ('adabst', AdaBoostClassifier(n_estimators=100, random_state=42)),
    ('gbc', GradientBoostingClassifier(
        n_estimators=50,
        max_depth=8,
        learning_rate= 0.1
        ))
    ]
vote_clf = VotingClassifier(
    estimators=base_estimator,
    voting='soft')
vote_clf.fit(X_trn, y_trn)
y_vote_pred = vote_clf.predict(X_tst)
y_vote_prob = vote_clf.predict_proba(X_tst)
# Plot confusion matrix
con_mat=confusion_matrix(y_tst,y_vote_pred)
#print(con_mat)
class_name=['Zone 1','Zone 2','Zone 3','Zone 4']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[4, 4])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('vote',fontweight = 'bold',fontsize = 10)
plt.show()
# ROC for vote classifier
ROC_Curve_all(y_true, y_prob_pred=y_vote_prob,clf_name='vote')
precision_recall_all(y_true, y_prob = y_vote_prob, clf_name='vote')
############### ROC For all clf for performance comparision ###################
clf_name = ['SVM','rf','adaboost','GBC','stack','vote']
y_prob = [y_svm_prob, y_rf_prob,y_adabst_prob, y_gbc_prob, y_stc_prob, y_vote_prob]
color = ['red','green', 'blue','black','orange','maroon']
# Plot ROC
for i in range(6):
    ROC_Curve_avg(
        y_true, 
        y_prob_pred=y_prob[i], 
        name = clf_name[i],
        color=color[i]
        )
plt.show()
#plot precision recall 
for i in range(6):
    precision_recall_avg(
        y_true, 
        y_prob=y_prob[i], 
        name = clf_name[i], 
        color = color[i])
plt.show()



