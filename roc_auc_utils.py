import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

N_CLASSES = 4


def _compute_roc(y_true, y_prob_pred):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_prob_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= N_CLASSES
    fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    return fpr, tpr, roc_auc


def ROC_Curve_avg(y_true, y_prob_pred, name, color):
    fpr, tpr, roc_auc = _compute_roc(y_true, y_prob_pred)
    area = roc_auc['micro']
    plt.plot(fpr['micro'], tpr['micro'],
             label=f'{name} ROC (area = {"%0.04f" % area})',
             color=color, linestyle='-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim(-0.01, 1)
    plt.ylim((0, 1.05))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')


def ROC_Curve_all(y_true, y_prob_pred, clf_name):
    fpr, tpr, roc_auc = _compute_roc(y_true, y_prob_pred)
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC (area ={0:04f})'.format(roc_auc['micro']),
             color='deeppink', linestyle=':', linewidth=2)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'maroon'])
    for i, c in zip(range(N_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=c, lw=2,
                 label='ROC Zone {0} (area = {1:0.4f})'.format(i + 1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title(clf_name)
    plt.show()


def precision_recall_all(y_true, y_prob, clf_name):
    precision, recall, average_precision = dict(), dict(), dict()
    for i in range(N_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])
    precision['micro'], recall['micro'], _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    average_precision['micro'] = average_precision_score(y_true, y_prob, average='micro')
    plt.plot(recall['micro'], precision['micro'],
             label='Average (area = {0:0.3f})'.format(average_precision['micro']))
    for i in range(N_CLASSES):
        plt.plot(recall[i], precision[i], lw=2,
                 label='Zone {0} (area {1:0.4f})'.format(i + 1, average_precision[i]))
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc='best')
    plt.title(clf_name)
    plt.show()


def precision_recall_avg(y_true, y_prob, name, color):
    precision, recall, average_precision = dict(), dict(), dict()
    for i in range(N_CLASSES):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])
    precision['micro'], recall['micro'], _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    average_precision['micro'] = average_precision_score(y_true, y_prob, average='micro')
    area = average_precision['micro']
    plt.plot(recall['micro'], precision['micro'],
             label=f'{name} (area = {"%0.04f" % area})')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(loc='best')
