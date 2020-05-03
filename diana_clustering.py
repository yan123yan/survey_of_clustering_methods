import numpy as np

class DianaClustering(object):

    def __init__(self, ds, k):
        self.clusters = [ds.data[:,:2].tolist()] #只考虑二维数据 花萼长度,花萼宽度
        self._k = k  # 结束条件
        self.num_k = 0

    def dist(self, point1, point2):
        return np.sqrt(np.sum(np.square(point1 - point2)))

    def find_max_dist_index(self, clusters):
        max_index = -1
        max = 0
        print(clusters)
        for n in range(len(clusters)):
            for j in range(len(clusters[n])):
                for i in range(len(clusters[n][j]) - 1):
                    #每个clusters[i]是一个 np array
                    object = clusters[n][j][i]
                    compared_object = clusters[n][j][i+1]
                    dist = self.dist(object,compared_object)
                    if dist > max:
                        max = dist
                        max_index = n
        return max_index

    def find_max_dist_point(self,clusters):
        #这层是clusters的个数
        for n in range(len(clusters)):
            #这层是循环一次的
            splinter_group = []
            old_party = []
            for j in range(len(clusters[n])):
                avg_max = 0
                max_index = -1
                # 这层是循环每个点的
                for i in range(len(clusters[n][j])):
                    #每个clusters[i]是一个 np array
                    object = clusters[n][j][i]
                    max_dist = 0
                    for k in range(len(clusters[n][j])):
                        compared_object = clusters[n][j][k]
                        dist = self.dist(object,compared_object)
                        max_dist = max_dist + dist
                    #计算1：1~500， 2：1~500的平均距离
                    avg_dist = max_dist / len(clusters[n][j])
                    if avg_dist > avg_max:
                        max_index = i
                splinter_group.append(clusters[n][j][max_index])
                for e in range(len(clusters[n][j])):
                    if e != max_index:
                        old_party.append(clusters[n][j][e])

                for h in range(len(old_party)):
                    max_d_2_sg = 0
                    max_d = 100
                    for q in range(len(splinter_group)):
                        d = self.dist(old_party[h],splinter_group[q])
                        if d < max_d_2_sg:
                            compared_sg_point = splinter_group[q]
                            max_d = d

                    min_d = 100
                    for p in range(len(old_party)):

                        if p != h:
                            c_d = self.dist(old_party[h],old_party[p])
                            if c_d >= max_d:
                                splinter_group.append(old_party[p])

            clusters.append(splinter_group)
            clusters.append(old_party)
            clusters.pop(0)
            return clusters

    def compute(self):
        while self.num_k < self._k:
            self.num_k = len(self.clusters)
            self.find_max_dist_index(self.clusters)
            self.clusters = self.find_max_dist_point(self.clusters)





