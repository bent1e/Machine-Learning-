import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def MDS(data):
    # m是样本数目，n是特征数目（维数）
    m, n = data.shape
    # 初始化距离矩阵D
    dist = np.zeros((m, m))
    # 初始化D的第i行的平方和的平均值
    disti = np.zeros(m)
    # 初始化D的第j列的平方和的平均值
    distj = np.zeros(m)
    # 初始化内积矩阵B
    B = np.zeros((m, m))
    for i in range(m):
        # 计算距离矩阵D（对应公式（2））
        dist[i] = np.sum(np.square(data[i] - data), axis=1).reshape(1, m)
    for i in range(m):
        # 计算D的第i行的平方和的平均值（对应公式（9）和（14））
        disti[i] = np.mean(dist[i, :])
        # 计算D的第j列的平方和的平均值（对应公式（10）和（15））
        distj[i] = np.mean(dist[:, i])
    # 计算矩阵的迹
    print(dist)
    distij = np.mean(dist)
    print(distij)
    for i in range(m):
        for j in range(m):
            # 计算B（i，j）（对应公式（16））
            B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
    # 特征值分解（对应公式（17））jk,
    lamda, V = np.linalg.eigh(B)
    index = np.argsort(-lamda)[:10]
    # 选取其中非零特征值
    diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:10]))
    # 非零特征值对应的特征向量
    V_selected = V[:, index]
    # 计算Z矩阵（对应公式（18））
    Z = V_selected.dot(diag_lamda)
    return Z

if __name__ == '__main__':
    # 读取数据集
    iris = load_iris()
    data = iris.data
    # MDS
    clf1 = MDS(data)
    # 查看效果
    plt.scatter(clf1[:,0],clf1[:,1],c=iris.target)
    # plt.scatter(data[:, 0], data[:, 1], c=iris.target)
    plt.title('MDS')
    plt.show()
