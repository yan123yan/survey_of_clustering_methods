import math
import matplotlib.pyplot as plt

class AgnesClustering(object):

    def __init__(self, ds,k):
        dd = ds.data[:,:2] #只考虑二维数据 花萼长度,花萼宽度
        self.clusters = []
        self._k = k #结束条件
        self.num_k = 100

        self._iris = []
        for i in dd:
            c = (i[0],i[1])
            self._iris.append(c)

    def plot(self, C):
        colValue = ['r', 'b', 'g', 'y', 'c', 'k', 'm']
        shapes = ['^', 's', 'v','p','8']
        for i in range(len(C)):
            coo_X = []  # x坐标列表
            coo_Y = []  # y坐标列表
            for j in range(len(C[i])):
                coo_X.append(C[i][j][0])
                coo_Y.append(C[i][j][1])
            plt.scatter(coo_X, coo_Y, marker=shapes[i], color=colValue[i % len(colValue)])

        plt.legend(loc='upper right')
        plt.show()


    # 计算欧几里得距离,a,b分别为两个元组
    def dist(self, a, b):
        return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

    # dist_min
    def dist_min(self, Ci, Cj):
        return min(self.dist(i, j) for i in Ci for j in Cj)

    # dist_max
    def dist_max(self, Ci, Cj):
        return max(self.dist(i, j) for i in Ci for j in Cj)

    # dist_avg
    def dist_avg(self, Ci, Cj):
        return sum(self.dist(i, j) for i in Ci for j in Cj) / (len(Ci) * len(Cj))

    # 找到距离最小的下标
    def find_Min(self, M):
        min = 1000
        x = 0
        y = 0
        for i in range(len(M)):
            for j in range(len(M[i])):
                if i != j and M[i][j] < min:
                    min = M[i][j]
                    x = i
                    y = j
        return (x, y, min)

    def compute(self):
        # 初始化C和M
        C = []
        M = []
        for i in self._iris:
            Ci = []
            Ci.append(i)
            C.append(Ci)
        for i in C:
            Mi = []
            for j in C:
                Mi.append(self.dist_avg(i, j))
            M.append(Mi)
        q = len(self._iris)
        # 合并更新
        while q > self._k:
            x, y, min = self.find_Min(M)
            C[x].extend(C[y])
            C.remove(C[y])
            M = []
            for i in C:
                Mi = []
                for j in C:
                    Mi.append(self.dist_avg(i, j))
                M.append(Mi)
            q -= 1

        self.plot(C)