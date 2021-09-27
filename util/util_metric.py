# ---encoding:utf-8---
# @Time : 2020.12.26
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : util_metric.py


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import numpy as np
import torch


def caculate_metric(pred_y, labels, pred_prob):
    # print('labels', labels) # [n_sample, num_class]
    # print('pred_y', pred_y) # [n_sample, num_class]
    # print('pred_prob', pred_prob) # [n_sample, num_class]

    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # print('tp\tfp\ttn\tfn')
    # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    labels = labels.cpu()
    pred_prob = pred_prob.cpu()
    labels = labels.numpy().tolist()
    pred_prob = pred_prob.numpy().tolist()

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC, tp, fp, tn, fn])

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data


# draw ROC
def ROC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.title('Receiver operating characteristic example', fontdict={'weight': 'normal', 'size': 30})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 30})
    plt.show()


# draw PRC
def PRC(recall, precision, AP):
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(AP))
    plt.show()
