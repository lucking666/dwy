import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import math
from linear_regression_std import tls, ls
import random
# 定义评价指标
def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))
def mape(y_true, y_pred):
    return sum(np.abs((y_true - y_pred) / y_true)) / len(y_true)

items=['F2', 'F3', 'F5', 'F6', 'F9']

# 加载数据
# 加载数据
data_all = pd.read_csv('dataset.csv')
data = data_all[['F2', 'F3', 'F5', 'F6', 'F9','cyclelife']]  # 注意特征与feature_remain保持一致!!!
_class = [0] * 41 + [1] * 43 + [2] * 40
data['class']=_class
_xita=[0]*124
data['xita']=_xita
# print(data)

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
n = 20 # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
s = 1      # 分割数据的次数（对数据进行随机排序的次数）
m = 1  # 对于每次分割得到的训练集，生成m次噪声


med_tls_rmse = []
med_ls_rmse = []
med_tls_em_rmse = []
med_ls_em_rmse = []

# 调整训练集的大小
for j in range(n):  # 调整噪声大小
    tls_rmse = []
    ls_rmse = []
    tls_em_rmse = []
    ls_em_rmse = []
    copy_data = copy.deepcopy(data_x)


    times=[]
    times.append(random.randint(0, 10) * j * 0.05)
    times.append(random.randint(0, 10) * j * 0.05)
    times.append(random.randint(0, 10) * j * 0.05)

    print('j:',j)
    print('times[]=:', times)

    for p in range(s): # 分割数据
        # 划分训练集与测试集
        np.random.seed(p)
        # random_datax = copy_data.reindex(np.random.permutation(copy_data.index))  # 随机排序
        random_datax=copy_data
        # 按照每个电池批次进行划分之后再合并
        X_data_1=random_datax.iloc[:41,:]
        X_data_2 = random_datax.iloc[41:84, :]
        X_data_3 = random_datax.iloc[84:, :]
        X_train1 = X_data_1.iloc[:N_train[0],:]
        X_test1 = X_data_1.iloc[N_train[0]:, :]
        X_train2 = X_data_2.iloc[:N_train[1], :]
        X_test2 = X_data_2.iloc[N_train[1]:, :]
        X_train3 = X_data_3.iloc[:N_train[2], :]
        X_test3 = X_data_3.iloc[N_train[2]:, :]

        data_train_random = pd.concat([pd.concat([X_train1,X_train2]),X_train3])
        X_test_random = pd.concat([pd.concat([X_test1, X_test2]), X_test3])

        data_train_random['cyclelife'] = np.log10(np.array(data_train_random['cyclelife']))#dataframe
        X_test_random['cyclelife'] = np.log10(np.array(X_test_random['cyclelife']))#dataframe
        # print('data_train_random is :',data_train_random )
        # print('X_test_random is :',X_test_random)
        Y_test_random=np.array(X_test_random['cyclelife']).reshape(-1,1)
        # print(type(Y_test_random))
        # print(Y_test_random.shape)
        # print('Y_test_random is :',Y_test_random)

        for k in range(m):  # 生成m次噪声

            X_train = copy.deepcopy(data_train_random)
            # Y_train = copy.deepcopy(data_train_random.iloc)
            X_test_random = X_test_random.iloc[:, 0:5]
            X_test = copy.deepcopy(X_test_random)  # dataframe
            Y_test = copy.deepcopy(Y_test_random)
            # print('X_test',X_test)
            # print('Y_test',Y_test)



            X_train_noise = copy.deepcopy(X_train).values


            # 转换数据类型（前面使用DataFrame是因为之前要进行特征选择）
            x_train = X_train_noise
            Y_train_noise = np.array(X_train_noise[:, 5]).reshape(-1, 1)
            x_test = X_test.values


            # print('x_train is :',x_train)
            # print('type x_train is:',type(x_train))
            # print('Y_train_noise is :', Y_train_noise)
            # print('type Y_train_noise is:', type(Y_train_noise))
            # print('x_test is :', x_test)
            # print('type x_test is:', type(x_test))
            # print('Y_test is :', Y_test)
            # print('type Y_test is:', type(Y_test))

            #   不加噪声
            print('x_train:',x_train)
            print('Y_train_noise:', Y_train_noise)

            # 总体最小二乘
            W_tls, b_tls, = tls(x_train[:,0:5], Y_train_noise)
            y_pred_tls = np.dot(x_test, W_tls) + b_tls
            tls_rmse.append(rmse(Y_test, y_pred_tls))
            # 最小二乘
            W_ls, b_ls, = ls(x_train[:,0:5], Y_train_noise)
            y_pred_ls = np.dot(x_test, W_ls) + b_ls
            ls_rmse.append(rmse(Y_test, y_pred_ls))

        print('tls_rmse:', tls_rmse)
        print('tls_em_rmse:', tls_em_rmse)

    print('目前进度：{:.0%}'.format((j + 1) / n))
    # med_tls_rmse.append(np.median(tls_rmse))
    med_ls_rmse.append(np.median(ls_rmse))
    med_tls_rmse.append(np.median(tls_rmse))

#
# 画图tls
plt.plot(med_tls_rmse)
plt.plot(med_ls_rmse)
plt.legend(['TLS', 'LS'])  #
plt.xlabel('experiments {}'.format(n*s*m))
plt.ylabel('RMSE')
plt.show()

