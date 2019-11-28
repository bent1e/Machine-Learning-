import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits

# 距离矩阵
'''
输入：数据集，大小（ m * m）
输出：距离矩阵
'''
def Distance(test):
    leng = len(test)
    resmat = np.zeros([leng, leng], np.float32)
    for i in range(leng):
        for j in range(leng):
            resmat[i, j] = np.linalg.norm(test[i] - test[j]) # 向量范数（欧式距离）
    return resmat

# MDS算法
'''
输入：数据集（test）、降维的维数（target）
输出：降维后的矩阵(Z)
'''
def MDS(test , target):

    # 初始化距离矩阵D
    length = len(test)
    if (target > length):
        target = length
    D = Distance(test)
    # m是样本数目，n是特征数目（维数）
    m, n = D.shape
    dist = np.zeros((m, m))
    # 初始化D的第i行的平方和的平均值
    disti = np.zeros(m)
    # 初始化D的第j列的平方和的平均值
    distj = np.zeros(m)
    # 初始化内积矩阵B
    B = np.zeros((m, m))
    for i in range(m):
        # 计算距离矩阵D
        dist[i] = np.sum(np.square(D[i] - D), axis=1).reshape(1, m)
    for i in range(m):
        # 计算D的第i行的平方和的平均值
        disti[i] = np.mean(dist[i, :])
        # 计算D的第j列的平方和的平均值
        distj[i] = np.mean(dist[:, i])
    # 计算矩阵的迹
    print(dist)
    distij = np.mean(dist)
    print(distij)
    for i in range(m):
        for j in range(m):
            # 计算B（i，j）
            B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
    # 特征值分解
    lamda, V = np.linalg.eigh(B)
    index = np.argsort(-lamda)[:target]
    # 选取其中非零特征值
    diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:target]))
    # 非零特征值对应的特征向量
    V_selected = V[:, index]
    # 计算Z矩阵
    Z = V_selected.dot(diag_lamda)
    return Z


# Dijkstra算法
'''
输入：数据集（test）大小 （m * m）、最短路径的起始点（start）范围 0 到 m-1
输出：降维后的矩阵
'''
def usedijk(test, start):
    count = len(test)
    col = test[start].copy()
    rem = count - 1
    while rem > 0:
        i = np.argpartition(col, 1)[1]
        length = test[start][i]
        for j in range(count):
            if test[start][j] > length + test[i][j]:
                test[start][j] = length + test[i][j]
                test[j][start] = test[start][j]
        rem -= 1
        col[i] = float('inf')


# isomap 算法
'''
输入：数据集（test）大小 （m * m）、降维的维数（target）、k近邻参数（k）
输出：降维后的矩阵
'''

def Isomap(test, target, k):
    inf = float('inf')
    count = len(test)
    if k >= count:
        raise ValueError('K is too large')
    mat_distance = Distance(test)
    knear = np.ones([count, count], np.float32)
    for idx in range(count):
        topk = np.argpartition(mat_distance[idx], k)[:k + 1]
        knear[idx][topk] = mat_distance[idx][topk]
    for idx in range(count):
        usedijk(knear, idx)
    return MDS(knear, target)


if __name__ == '__main__':
    data = pd.read_csv('./train.csv', header=0)
    imgs = data.iloc[1:, 1:].values
    test = imgs[:500]
    print(test)
    labels = data.iloc[1:, 0].values
    print('开始降维.....')
    outcome = Isomap(test, 2, 3)
    plt.scatter(outcome[:,0].flatten(),outcome[:,1].flatten())
    plt.savefig('./Isomap.png')