import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.model_selection import LeaveOneOut
import math
import random


# from libs.evaluation_indicators import rmse
from linear_regression_std import tls, ls
# from libs.stepwise import cv_stepwise

def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))
# 加载数据
data_all = pd.read_csv('dataset.csv')
data = data_all[['F2', 'F3', 'F5', 'F6', 'F9','cyclelife']]  # 注意特征与feature_remain保持一致!!!
_class = [0] * 41 + [1] * 43 + [2] * 40
data['class']=_class
_xita=[0]*124
data['xita']=_xita
data['cyclelife'] = np.log10(np.array(data['cyclelife']))
items=['F2','F3','F5','F6','F9']

def add_em(X, Y,flag):
    _rmse=[]
    W = np.ones((5, 1))
    X_test=X[:,0:5]
    Y_test=Y
    b = [[1]]
    i=0

    xxx=1000.0
    while i<1:#xxx>10
        i+=1
        X_dataframe=pd.DataFrame(X,columns=['F2','F3','F5','F6','F9','cyclelife','class','xita'])
        Y_predict = np.dot(X[:, 0:5], W) + b
        xita = Y_predict - Y
        X_dataframe['xita']=xita
        for index in range(3):
            lamuda = 1 / math.sqrt(X_dataframe[X_dataframe['class'] == index]['xita'].var())
            for item in items:
                X_dataframe.loc[X_dataframe['class'] == index, item] =X_dataframe[X_dataframe['class'] == index][item] * lamuda
            X_dataframe.loc[X_dataframe['class'] == index, 'cyclelife'] = X_dataframe[X_dataframe['class'] == index]['cyclelife'] * lamuda
        X_ = X_dataframe[['F2', 'F3', 'F5', 'F6', 'F9']].values
        Y_ = X_dataframe['cyclelife'].values.reshape(-1,1)
        if flag=='tls':
            W, b = tls(X_, Y_)
        if flag=='ls':
            W, b =ls(X_, Y_)

        y_pred_tls_em = np.dot(X_test, W) + b
        # print('y_pred_tls_em :',y_pred_tls_em)
        # print('Y_test :', Y_test)
        _rmse.append(rmse(Y_test, y_pred_tls_em))
        # xxx=rmse(Y_test, y_pred_tls_em)
        # print("rmse is :",xxx)

    # print('_rmse is :',_rmse)


    # plt.plot(_rmse)
    # plt.show()

    return W,b
# 数据集划分
# data1 = data.iloc[:41, ]  # DataFrame切片
# data2 = data.iloc[41:84, ]
# data3 = data.iloc[84:, ]

# 选择数据集
data_x = copy.deepcopy(data)
N_train = []
N_train.append(round(41 * 0.9))  # 训练集比例
N_train.append(round(43 * 0.9))
N_train.append(round(40 * 0.9))

# n = 20  # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
# s = 100  # 分割数据的次数（对数据进行随机排序的次数）
# m = 50  # 对于每次分割得到的训练集，生成m次噪声
n = 20  # 最大噪声水平：times =19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
s = 1  # 分割数据的次数（对数据进行随机排序的次数）
m = 1  # 对于每次分割得到的训练集，生成m次噪声

med_tls_rmse = []
med_ls_rmse = []
med_tls_rmse_em = []
med_ls_rmse_em = []
for j in range(n):  # 调整噪声大小
    tls_rmse = []
    ls_rmse = []
    tls_rmse_em = []
    ls_rmse_em = []
    copy_data = copy.deepcopy(data_x)
    times = j * 0.05
    for p in range(s): # 分割数据
        # 划分训练集与测试集
        np.random.seed(p)
        random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
        X_train_random = random_datax.iloc[:N_train, :]
        Y_train_random = np.array(random_datax.iloc[:N_train, 5]).reshape(-1, 1)
        X_test_random = random_datax.iloc[N_train:, 0:5]
        Y_test_random = np.array(random_datax.iloc[N_train:, 5]).reshape(-1, 1)
        # print('X_train_random',X_train_random)
        # print(type(X_train_random))
        # print('Y_train_random',Y_train_random)
        # print(type(Y_train_random))
        # print('X_test_random',X_test_random)
        # print(type(X_test_random))
        # print('Y_test_random',Y_test_random)
        # print(type(Y_test_random))

        for k in range(n):  # 生成噪声
            X_train = copy.deepcopy(X_train_random)
            Y_train = copy.deepcopy(Y_train_random)
            X_test = copy.deepcopy(X_test_random)
            Y_test = copy.deepcopy(Y_test_random)
            # standard_X = np.std(X_train, axis=0)#.reshape(-1, 1)
            # standard_Y = np.std(Y_train, axis=0)#.reshape(-1, 1)
            # noise_X = np.random.randn(X_train.shape[0], X_train.shape[1]) # 对于X，先生成一个噪声矩阵
            # noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1) # 对于y，直接根据其标准差生成噪声向量

            #加入噪声
            X_train_noise = copy.deepcopy(X_train)
            Y_train_noise = copy.deepcopy(Y_train)
            # for i in range(X_train.shape[1]):
            #     noise_X[:, i] *= (times * standard_X[i])   #根据每个特征的标准差生成噪声
            #     X_train_noise.values[:, i] += noise_X[:, i]
            # Y_train_noise += noise_Y

            # 转换数据类型（前面使用DataFrame是因为之前要进行特征选择）
            x_train = X_train_noise.values
            x_test = X_test.values

            # 总体最小二乘
            W_tls, b_tls, = tls(x_train[:,0:5], Y_train_noise)
            y_pred_tls = np.dot(x_test, W_tls) + b_tls
            tls_rmse.append(rmse(Y_test, y_pred_tls))

            W_tls_em, b_tls_em, = add_em(x_train, Y_train_noise,'tls')
            y_pred_tls_em = np.dot(x_test, W_tls_em) + b_tls_em
            tls_rmse_em.append(rmse(Y_test, y_pred_tls_em))

            # 最小二乘
            W_ls, b_ls, = ls(x_train[:,0:5], Y_train_noise)
            y_pred_ls = np.dot(x_test, W_ls) + b_ls
            ls_rmse.append(rmse(Y_test, y_pred_ls))

            W_ls_em, b_ls_em, = add_em(x_train, Y_train_noise,'ls')
            y_pred_ls_em = np.dot(x_test, W_ls_em) + b_ls_em
            ls_rmse_em.append(rmse(Y_test, y_pred_ls_em))

    print('目前进度：{:.0%}'.format((j+1)/n))
    med_tls_rmse.append(np.median(tls_rmse))
    med_ls_rmse.append(np.median(ls_rmse))
    med_tls_rmse_em.append(np.median(tls_rmse_em))
    med_ls_rmse_em.append(np.median(ls_rmse_em))


# # 画图
plt.plot(med_tls_rmse)
plt.plot(med_tls_rmse_em)
plt.legend(['TLS', 'TLS_EM']) #
plt.xlabel('the number of experiments')
plt.ylabel('RMSE')
plt.show()

plt.plot(med_ls_rmse)
plt.plot(med_ls_rmse_em)
plt.legend(['LS', 'LS_EM']) #
plt.xlabel('the number of experiments')
plt.ylabel('RMSE')
plt.show()

# 保存rmse值
# tls_ls_rmse = np.empty(shape=(n, 2))
# tls_ls_rmse[:, 1] = med_tls_rmse
# tls_ls_rmse[:, 2] = med_ls_rmse
# df = pd.DataFrame(tls_ls_rmse)
# df.to_csv('noise_level_100.csv', index=False, header=False)


