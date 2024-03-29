# encoding=utf-8
# @Author: ZhuZhiPeng
# @Date:   07-10-19
# @Email:  bentle@163.com


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time

def softMax(x,y,alpha,number):
    theta=np.ones((10,785)) #初始theta矩阵  
    for i in range(number):  #迭代
        k=np.random.randint(0,len(X_train)) #从样本中随机挑选一个做优化
        x_ = x[k].reshape(785, 1)
        theta_T_x = np.dot(theta, x_)
        e_theta_T_x = np.exp(theta_T_x)
        denominator = e_theta_T_x.sum() #分母
        numerator = e_theta_T_x # 分子
        fraction = numerator / denominator
        y_vector = np.where(np.arange(10).reshape(10, 1) == y[k], 1, 0)# 按类别做编号处理（非1即0）
        gradient = (fraction - y_vector) * x[k]
        # gradient = (fraction - y_vector) * x[k] + 0.01*theta
        theta -= alpha * gradient
    return theta



if __name__ == '__main__':
    number=100000
    alpha=0.01
    # MINIST手写数字集
    data = pd.read_csv('./train.csv', header=0)
    # 数据预处理
    data.insert(1, 'basic', 1)
    imgs = data.iloc[1:, 1:].values
    labels = data.iloc[1:, 0].values
    # 防止exp计算数据溢出，做标准化处理
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    imgs = min_max_scaler.fit_transform(imgs)
    # 选取训练集、测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    # 训练
    time_1 = time.time()
    theta = softMax(X_train, Y_train, alpha=alpha,number=number)
    time_2 = time.time()
    print('训练消耗 ', time_2 - time_1, ' 秒', '\n','迭代次数',number,'次','\n','学习速率', alpha)
    # 测试
    predict = np.dot(theta, X_test.T)
    predict_sort = predict.argmax(axis=0)
    num_of_wrong = (predict_sort != Y_test).sum()
    print('前20个正确值：', Y_test.T[:20])
    print('前20个预测值：',predict_sort[:20])
    print("错误的个数: ", num_of_wrong)
    print("正确率为: {0}%".format((len(Y_test) - num_of_wrong) / len(Y_test) * 100))