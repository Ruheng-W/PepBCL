# -*- coding: utf-8 -*-
# @Time    : 2021/5/7 16:50
# @Author  : WANG Ruheng
# @Email   : blwangheng@163.com
# @IDE     : PyCharm
# @FileName: data_loader_protBert.py

from util import util_dim_reduction

import numpy as np
import matplotlib.pyplot as plt

def dimension_reduction(repres_list, label_list, epoch):
    print('t-SNE')
    title = 'Samples Embedding t-SNE Visualisation, Epoch[{}]'.format(epoch)
    util_dim_reduction.t_sne(title, repres_list, label_list, None, 2)
    # print('PCA')
    # title = 'Samples Embedding PCA Visualisation, Epoch[{}]'.format(epoch)
    # util_dim_reduction.pca(title, repres_list, label_list, None, 4)


def penultimate_feature_visulization(repres_list, label_list, epoch):
    X = np.array(repres_list)  # [num_samples, n_components]
    data_index = label_list
    data_label = None
    class_num = 4

    # draw
    title = 'Learned Feature Visualization, Epoch[{}]'.format(epoch)
    font = {"color": "darkred", "size": 13, "family": "serif"}
    plt.style.use("default")

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X)):
            plt.annotate(data_label[i], xy=(X[:, 0][i], X[:, 1][i]),
                         xytext=(X[:, 0][i] + 1, X[:, 1][i] + 1))
            # X, Y is the coordinate to be marked, and 'xytext' is the corresponding label coordinate
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    plt.show()
