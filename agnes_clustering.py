import numpy as np
import matplotlib.pyplot as plt

class AgnesClustering(object):

    def __init__(self, ds,k):
        self._iris = ds.data[:,:2] #只考虑二维数据 花萼长度,花萼宽度
        self.clusters = []
        self._k = k #结束条件
        self.num_k = 100

        #print(self._iris.shape[0])

    def find_min_dist(self, ls_dists):
        ls_mins = []
        ls_indexs = []
        for ls_dist in ls_dists:
            min = 100
            index = -1
            for i in range(len(ls_dist)):
                #print("aaa",ls_dists[i])
                if ls_dist[i] < min:
                    min = ls_dist[i]
                    index = i
            ls_mins.append(min)
            ls_indexs.append(index)
        return ls_mins, ls_indexs

    def compute(self):
        # 将整体拆分为每个独立
        #为每个样本分配一个聚类簇
        for ind in range(len(self._iris)):
            self.clusters.append([ind])

        while self.num_k > self._k:

            global_index = []

            # 计算任意集合之间的距离
            ls_dists = []
            #循环每个集合
            for i in range(len(self.clusters)):
                set_value = self._iris[self.clusters[i]][0]
                ls_dist = []
                for j in range(len(self.clusters)):
                    compared_value = self._iris[self.clusters[j]][0]
                    dist = np.sqrt(np.sum(np.square(np.array(set_value) - np.array(compared_value))))
                    if dist == 0:
                        dist = 100
                    ls_dist.append(dist)
                ls_dists.append(ls_dist)
            #获取最短距离的数据的下标
            ls_mins, ls_indexs = self.find_min_dist(ls_dists)
            #合并簇
            new_list = []
            for q in range(len(ls_indexs)):

                if q in global_index:
                    continue

                #找出最近邻对象的index
                compared_object_index = -1
                for a in range(len(self.clusters)):
                    if ls_indexs[q] in self.clusters[a]:
                        compared_object_index = a
                        break

                # 找出插入对象的index
                object_index = -1
                for c in range(len(self.clusters)):
                    if q in self.clusters[c]:
                        object_index = c
                        break

                global_index.append(object_index)
                global_index.append(compared_object_index)
                new_list.append(self.clusters[object_index] + self.clusters[compared_object_index])

            print(new_list)
            print(len(new_list))
            self.clusters = new_list

            # 计算簇的数量
            self.num_k = len(self.clusters)
            print("num: ", self.num_k)
