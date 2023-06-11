import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.model_selection import LeaveOneOut
import math
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression   #调库

from linear_regression_std import tls, ls
def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))



x=[]
x.append(38)
x.append(35)
x.append(56)
for c in range(3):
    x[c]=x[c]/np.sum(x)

print(x)
#
# items=['a','b','c','d','e','f','xita','class']
# def add_em(X, Y,flag):
#     _rmse=[]
#     # 使用随机初始值
#     rd = np.random.RandomState()
#     # 随机浮点数
#     W = rd.random((5, 1))
#     b = rd.random((1, 1))
#     print('初始模型系数为：',W,b)
#     #
#     # # # 使用固定初始值
#     # W = np.ones((5, 1))
#     # b = [[1]]
#     i = 0
#     dis=1111
#
#
#     while i<4:# dis>1e-10
#
#         print(i)
#         X_dataframe = pd.DataFrame(X, columns=items)
#         Y_predict = np.dot(X[:, 0:5], W) + b
#         print('Y_predict',Y_predict)
#         xita = Y_predict - Y
#         print('xita',xita)
#         X_dataframe['xita']=xita
#         for index in range(3):
#             print('index',index)
#             # 使用零均值求标准差
#             # xitanum=X_dataframe[X_dataframe['class'] == index]['xita']
#             # _std=math.sqrt(sum([(x - 0) ** 2 for x in xitanum]) / len(xitanum))
#             # 直接求标准差
#             _std = X_dataframe[X_dataframe['class'] == index]['xita'].std()
#             # _std=2
#             print(_std)
#
#
#             lamuda = 1/_std
#             print('lamuda',lamuda)
#             #对数据集加权
#             for item in ['a','b','c','d','e']:
#                 X_dataframe.loc[X_dataframe['class'] == index, item] =X_dataframe[X_dataframe['class'] == index][item] * lamuda   #-np.log(lamuda)
#
#             print('输出每个类别的f',X_dataframe[X_dataframe['class'] == index]['f'])
#             X_dataframe.loc[X_dataframe['class'] == index, 'f'] = X_dataframe[X_dataframe['class'] == index]['f'] * lamuda   #-np.log(lamuda)
#         #加权数据几求模型系数
#         # print(X_dataframe)
#         X_ = X_dataframe[['a','b','c','d','e','f']].values
#         Y_ = X_dataframe['f'].values.reshape(-1,1)
#         print('X_',X_)
#         print('Y_', Y_)
#         lr = LinearRegression()
#         lr.fit(X_,Y_)
#         W_em = lr.coef_[0][0]  # 内置的一个变量，表示系数。输出是一个二维数组
#         b_em = lr.intercept_[0]
#
#         dis=np.linalg.norm(W_em-W)
#         W, b = W_em, b_em
#
#         #先查看迭代过程中的rmse
#         y_pred_tls_em = np.dot(X[:,0:5], W) + b
#         # print('y_pred_tls_em :',y_pred_tls_em)
#         # print('Y_test :', Y_test)
#         s=rmse(Y, y_pred_tls_em)
#         _rmse.append(rmse(Y, y_pred_tls_em))
#         # xxx=rmse(Y_test, y_pred_tls_em)
#         print("rmse is :",s)
#         i += 1
#     #
#     #
#     print('_rmse is :',_rmse)
#
#
#     plt.plot(_rmse)
#     plt.show()
#
#     return W,b
#
#
# # 加载数据
# data_all = pd.read_csv('dataset_.csv')
# X=data_all.values
# Y=X[:,5].reshape(-1,1)
# print(X)
# print(Y)
# add_em(X,Y,'ls')