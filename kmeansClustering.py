import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering(object):

    def __init__(self, ds, k=2):
        self._k = k
        self.dataset = ds
        self._iris = ds.data[:,:2] #只考虑二维数据 花萼长度,花萼宽度
        self.result = []
        self.times = 0
        self.max_times = 10
        self.ls_group_iris = []
        self.ls_center_points = []

    def get_min_index(self,dist_ls):
        min = 10
        index = -1
        for i in range(len(dist_ls)):
            if dist_ls[i] < min:
                min = dist_ls[i]
                index = i
        return index

    def plot(self,points,ls_group_iris):
        # 散点图
        colors=['red','blue','green']
        shapes=['^','s','v']
        for i in range(len(ls_group_iris)):
            np_group_iris = np.array(ls_group_iris[i])
            plt.scatter(np_group_iris[:, 0], np_group_iris[:, 1], color=colors[i],marker=shapes[i])
        #中心点
        print("points",points)
        for each in points:
            plt.scatter(each[0],each[1],color='black',s=90,marker='o')
        plt.xlabel(self.dataset.feature_names[0])
        plt.ylabel(self.dataset.feature_names[1])
        plt.legend(loc='upper left')
        plt.show()

    def itera(self,np_iris,center_points,ls_group_iris):
        for point in np_iris:
            ls = []
            #获取到每个中心点的距离
            for center_point in center_points:
                dist = np.sqrt(np.sum(np.square(point - center_point)))
                ls.append(dist)
                #print(dist)
            #比较每个中心点的距离长度
            ls_group_iris[self.get_min_index(ls)].append(point)
        print("ls_group_iris 信息：")
        print("group1 的数量：",str(len(ls_group_iris[0])))
        print("group2 的数量：", str(len(ls_group_iris[1])))
        #清空中心点位置，待更新
        center_points.clear()
        #计算每一组的平均值
        for each_group in ls_group_iris:
            #每个group有若干个点
            np_group = np.array(each_group)
            l = []
            l.append(np.sum(np_group[:,0])/len(each_group))
            l.append(np.sum(np_group[:,1])/len(each_group))
            mean_point = np.array(l)
            print(mean_point)
            #更新中心点
            center_points.append(mean_point)
        self.ls_group_iris = ls_group_iris
        #更新次数
        self.times = self.times + 1
        if self.times < self.max_times:
            #继续迭代
            ls_group_iris = [[] for _ in range(self._k)]
            self.itera(np_iris,center_points,ls_group_iris)
        else:
            self.ls_center_points = center_points
            return

    def compute(self):
        np_iris = np.array(self._iris)
        # random center 存放两个随机点
        center_points = [np_iris[np.random.randint(0, np_iris.shape[0])] for _ in range(self._k)]
        # print("random points: ", center_points)
        ls_group_iris = [[] for _ in range(self._k)]
        self.itera(np_iris,center_points,ls_group_iris)
        self.plot(self.ls_center_points,self.ls_group_iris)


'''
    def compute(self):
        np_iris = np.array(self._iris)
        #random center 存放两个随机中心点
        random_center = [np_iris[np.random.randint(0,np_iris.shape[0])] for _ in range(self._k)]
        print("random points: ",random_center)
        t = 0
        # for h in range(self._k):
        #     self.result.append([])
        while 1:
            if t == 0:
                #求随机点均值,先将随机点坐标转为list排序，获取其x值
                x_random_center = sorted([b.tolist()[0] for b in random_center])
                #print(x_random_center)
                #两两之间计算均值，存入该变量
                x_random_means = [(x_random_center[i] + x_random_center[i+1]) / 2 for i in range(len(x_random_center) - 1)]
                print(x_random_means)
                #迭代数据集进行分类
                if self._k == 2:
                    self.result.append(np_iris[np_iris[:, 0] < x_random_means[0]])
                    self.result.append(np_iris[np_iris[:, 0] >= x_random_means[0]])
                elif self._k == 3:
                    self.result.append(np_iris[np_iris[:, 0] < x_random_means[0]])
                    np_filter1 = np_iris[np_iris[:, 0] < x_random_means[1]]
                    self.result.append(np_filter1[np_filter1[:, 0] >= x_random_means[0]])
                    self.result.append(np_iris[np_iris[:, 0] >= x_random_means[1]])
                elif self._k == 4:
                    self.result.append(np_iris[np_iris[:, 0] < x_random_means[0]])
                    np_filter1 = np_iris[np_iris[:, 0] < x_random_means[1]]
                    self.result.append(np_filter1[np_filter1[:, 0] >= x_random_means[0]])
                    np_filter2 = np_iris[np_iris[:, 0] < x_random_means[2]]
                    self.result.append(np_filter2[np_filter2[:, 0] >= x_random_means[1]])
                    #self.result.append(np.where((np_iris[:, 0] < x_random_means[1]) & (np_iris[:,0] >= x_random_means[0])))
                    #self.result.append(np.where((np_iris[:, 0] < x_random_means[2]) & (np_iris[:, 0] >= x_random_means[1])))
                    self.result.append(np_iris[np_iris[:, 0] >= x_random_means[2]])
                print(self.result)
                #result为已经随机分好组的list
                t = t + 1

            #跳出第一次初始化
'''