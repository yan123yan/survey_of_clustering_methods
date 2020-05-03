import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from  sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch

class SK_Clustering(object):

    def __init__(self, ds):
        self._iris = ds.data[:,:2] #只考虑二维数据 花萼长度,花萼宽度

    def dbscan(self, eps=0.4, min_samples=9):
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        clustering.fit(self._iris)
        label_pred = clustering.labels_

        #绘制结果
        x0 = self._iris[label_pred == 0]
        x1 = self._iris[label_pred == 1]
        x2 = self._iris[label_pred == 2]
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.title('DBSCAN Clustering')
        plt.legend(loc=2)
        plt.show()

    def agnes(self, k=2):
        clustering = AgglomerativeClustering(linkage='ward',n_clusters=k)
        res = clustering.fit(self._iris)
        plt.figure()
        d0 = self._iris[clustering.labels_ == 0]
        plt.plot(d0[:, 0], d0[:, 1], 'r.')
        d1 = self._iris[clustering.labels_ == 1]
        plt.plot(d1[:, 0], d1[:, 1], 'go')
        d2 = self._iris[clustering.labels_ == 2]
        plt.plot(d2[:, 0], d2[:, 1], 'b*')
        plt.xlabel("Sepal.Length")
        plt.ylabel("Sepal.Width")
        plt.title("AGNES Clustering")
        plt.show()

    def kmeans(self, k=2):
        clustering = KMeans(n_clusters=k)  # 构造聚类器
        clustering.fit(self._iris)  # 聚类
        label_pred = clustering.labels_  # 获取聚类标签
        # 绘制k-means结果
        x0 = self._iris[label_pred == 0]
        x1 = self._iris[label_pred == 1]
        x2 = self._iris[label_pred == 2]
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.title('KMEANS Clustering')
        plt.legend(loc=2)
        plt.show()

    def plotReachability(self,data, eps):
        plt.figure()
        plt.plot(range(0, len(data)), data)
        plt.plot([0, len(data)], [eps, eps])
        plt.show()

    def optics(self,min_samples=9, xi =0.05, min_size=0.05):
        clustering = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size= min_size)
        clustering.fit(self._iris)
        label_pred = clustering.labels_
        x0 = self._iris[label_pred == 0]
        x1 = self._iris[label_pred == 1]
        x2 = self._iris[label_pred == 2]
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.title('OPTICS Clustering')
        plt.legend(loc=2)
        plt.show()


    def birch(self):
        clustering = Birch(n_clusters=3)
        clustering.fit(self._iris)
        label_pred = clustering.labels_

        x0 = self._iris[label_pred == 0]
        x1 = self._iris[label_pred == 1]
        x2 = self._iris[label_pred == 2]
        plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
        plt.xlabel('sepal length')
        plt.ylabel('sepal width')
        plt.title('OPTICS Clustering')
        plt.legend(loc=2)
        plt.show()