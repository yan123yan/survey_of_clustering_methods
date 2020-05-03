import numpy as np
import matplotlib.pyplot as plt
import math
import random

# visitlist类用于记录访问列表
# unvisitedlist记录未访问过的点
# visitedlist记录已访问过的点
# unvisitednum记录访问过的点数量
class Visitlist:
    def _init_(self, count=0):
        self.unvisitedlist=[i for i in range(count)]
        self.visitedlist=list()
        self.unvisitednum=count

    def visit(self, pointId):
        self.visitedlist.append(pointId)
        self.unvisitedlist.remove(pointId)
        self.unvisitednum -= 1

class DbscanClustering(object):

    def __init__(self, ds):
        dd = ds.data[:,:2] #只考虑二维数据 花萼长度,花萼宽度

    def dist(self, a, b):
        # 计算a,b两个元组的欧几里得距离
        return math.sqrt(np.power(a - b, 2).sum())

    def plot(self):
        pass

    def compute(self, dataset, eps, minPts):
        # numpy.ndarray的 shape属性表示矩阵的行数与列数
        nPoints = dataset.shape[0]
        # (1)标记所有对象为unvisited
        # 在这里用一个类vPoints进行买现
        vPoints = Visitlist(count=nPoints)
        # 初始化簇标记列表C,簇标记为 k
        k = -1
        C = [-1 for i in range(nPoints)]
        while (vPoints.unvisitednum > 0):
            # (3)随机上选择一个unvisited对象p
            p = random.choice(vPoints.unvisitedlist)
            # (4)标记p为visited
            vPoints.visit(p)
            # (5)if p的$\varepsilon$-邻域至少有MinPts个对象
            # N是p的$\varepsilon$-邻域点列表
            N = [i for i in range(nPoints) if self.dist(dataset[i], dataset[p]) <= eps]
            if len(N) >= minPts:
                # (6)创建个新簇C，并把p添加到C
                # 这里的C是一个标记列表，直接对第p个结点进行赋植
                k += 1
                C[p] = k
                # (7)令N为p的ε-邻域中的对象的集合
                # N是p的$\varepsilon$-邻域点集合
                # (8) for N中的每个点p'
                for p1 in N:
                    # (9) if p'是unvisited
                    if p1 in vPoints.unvisitedlist:
                        # (10)标记p’为visited
                        vPoints.visit(p1)
                        # (11) if p'的$\varepsilon$-邻域至少有MinPts个点，把这些点添加到N
                        # 找出p'的$\varepsilon$-邻域点，并将这些点去重添加到N
                        M = [i for i in range(nPoints) if self.dist(dataset[i], dataset[p1]) <= eps]
                        if len(M) >= minPts:
                            for i in M:
                                if i not in N:
                                    N.append(i)
                        # (12) if p'还不是任何簇的成员，把P'添加到C
                        # C是标记列表，直接把p'分到对应的簇里即可
                        if C[p1] == -1:
                            C[p1] = k
            # (15)else标记p为噪声
            else:
                C[p] = -1

        # (16)until没有标t己为unvisitedl内对象
        return C
