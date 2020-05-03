from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import kmeansClustering as kc
import kmedianClistering as kmdc
import AGNESClustering as ac
import diana_clustering as dc
import SK_clustering as skc

def plotData(dataset):
    #散点图
    plt.scatter(dataset.data[:,0],dataset.data[:,1],c=dataset.target)
    plt.xlabel(dataset.feature_names[0])
    plt.ylabel(dataset.feature_names[1])
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    dataset = load_iris()

    #获取数据的键名
    keys = dataset.keys()
    #获取数据的条数和维数
    n_samples, n_features = dataset.data.shape
    #获取数据的标签名
    target_names = dataset.target_names
    print("Keys: ", keys)  # ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
    print("Number of sample: ", n_samples)  # Number of sample: 150
    print("Number of feature: ", n_features) # Number of feature 4 #花萼长度,花萼宽度,花瓣长度,花瓣宽度
    print("Target names: ", target_names)  # ['setosa' 'versicolor' 'virginica']

    plotData(dataset)

    kmeans = kc.KMeansClustering(dataset,k=3)
    kmeans.compute()
    kmedian = kmdc.KMedianClustering(dataset,k=3)
    kmedian.compute()
    agnes = ac.AgnesClustering(dataset,k=3)
    agnes.compute()
    diana = dc.DianaClustering(dataset,k=3)
    diana.compute()
    sk = skc.SK_Clustering(dataset)
    sk.dbscan(eps=0.2, min_samples=5)
    sk.kmeans(k=3)
    sk.optics(min_samples=2, xi =0.05, min_size=0.15)
    sk.agnes(k=3)
    sk.birch()
