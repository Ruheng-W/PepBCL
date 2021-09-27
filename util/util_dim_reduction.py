# ---encoding:utf-8---
# @Time : 2021.01.28
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : util_dim_reduction.py


from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(title, data, data_index, data_label, class_num):
    X_pca = PCA(n_components=2).fit_transform(data)
    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")
    plt.figure()

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_pca)):
            plt.annotate(data_label[i], xy=(X_pca[:, 0][i], X_pca[:, 1][i]),
                         xytext=(X_pca[:, 0][i] + 0.00, X_pca[:, 1][i] + 0.00))
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    plt.show()


def t_sne(title, data, data_index, data_label, class_num):
    print('processing data')
    X_tsne = TSNE(n_components=2).fit_transform(data)  # [num_samples, n_components]
    print('processing data over')

    font = {"color": "darkred", "size": 13, "family": "serif"}
    # plt.style.use("dark_background")
    plt.style.use("default")

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_index, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', class_num))
    if data_label:
        for i in range(len(X_tsne)):
            plt.annotate(data_label[i], xy=(X_tsne[:, 0][i], X_tsne[:, 1][i]),
                         xytext=(X_tsne[:, 0][i] + 1, X_tsne[:, 1][i] + 1))
    plt.title(title, fontdict=font)

    if data_label is None:
        cbar = plt.colorbar(ticks=range(class_num))
        cbar.set_label(label='digit value', fontdict=font)
        plt.clim(0 - 0.5, class_num - 0.5)
    plt.show()


if __name__ == '__main__':
    digits = load_digits()

    print('data', digits.data)
    print('target', digits.target)
    print('data.shape', digits.data.shape)  # [n_samples, n_features]
    print('target.shape', digits.target.shape)  # [n_samples]

    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
    X_pca = PCA(n_components=2).fit_transform(digits.data)

    font = {"color": "darkred", "size": 13, "family": "serif"}

    # plt.style.use("dark_background")
    plt.style.use("default")

    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("t-SNE", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("PCA", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.tight_layout()
    plt.show()
