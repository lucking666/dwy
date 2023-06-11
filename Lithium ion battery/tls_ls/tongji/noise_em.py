import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.model_selection import LeaveOneOut
import math
import sys
import random
from sklearn.preprocessing import StandardScaler

# from libs.evaluation_indicators import rmse
from linear_regression_std import tls, ls
# from libs.stepwise import cv_stepwise

def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))

items=['F2','F3','F5','F6','F9']

#tls和em算法结合
def add_em(X, Y,flag,x_test,Y_test):
    X_test=x_test
    Y_test=Y_test
    rmse_train = []
    rmse_test = []
    # # 使用随机初始值
    # rd = np.random.RandomState()
    # # 随机浮点数
    # W = rd.random((5, 1))
    # b = rd.random((1, 1))

    # 使用固定初始值
    W = np.ones((5, 1))
    b = [[1]]
    i = 0
    dis=1111
    # print('w:',W)
    # print('b',b)

    while i<30:# dis>1e-10

        # print(i)
        X_dataframe = pd.DataFrame(X, columns=['F2', 'F3', 'F5', 'F6', 'F9', 'cyclelife', 'class', 'xita'])
        Y_predict = np.dot(X[:, 0:5], W) + b
        xita =Y-Y_predict
        # print('xita',xita)
        X_dataframe['xita']=xita
        lamuda=[]
        for index in range(3):
            _std = X_dataframe[X_dataframe['class'] == index]['xita'].std()
            lamuda.append(1/_std)
        #将标准差进行标准化
        for c in range(len(lamuda)):
            lamuda[c] = lamuda[c] / np.sum(lamuda)

        for index in range(3):
            #对数据集加权
            for item in items:
                X_dataframe.loc[X_dataframe['class'] == index, item] =X_dataframe[X_dataframe['class'] == index][item] * lamuda[index]
            X_dataframe.loc[X_dataframe['class'] == index, 'cyclelife'] = X_dataframe[X_dataframe['class'] == index]['cyclelife'] *lamuda[index]
        # if _ff == 0:
        #     break
        #加权数据几求模型系数
        X_ = X_dataframe[['F2', 'F3', 'F5', 'F6', 'F9']].values
        Y_ = X_dataframe['cyclelife'].values.reshape(-1,1)
        if flag=='tls':
            W_em, b_em = tls(X_, Y_)
        if flag=='ls':
            W_em, b_em =ls(X_, Y_)

        dis=np.linalg.norm(W_em-W)
        W, b = W_em, b_em

        #先查看迭代过程中的rmse

        rmse_train.append(rmse(Y, np.dot(X[:,0:5], W) + b))
        rmse_test.append(rmse(Y_test, np.dot(X_test, W) + b))
        # xxx=rmse(Y_test, y_pred_tls_em)
        # print("rmse is :",s)
        i += 1


    #
    plt.plot(rmse_train)
    plt.plot(rmse_test)
    plt.legend(['train', 'test'])  #
    plt.xlabel('loop')
    plt.ylabel('RMSE')
    plt.show()

    return W,b


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
n = 20  # 最大噪声水平：times=19*0.05，noise_Y = times * standard_Y * np.random.randn(Y_train.shape[0], 1)
s = 1       # 分割数据的次数（对数据进行随机排序的次数）
m = 1 # 对于每次分割得到的训练集，生成m次噪声

med_tls_rmse = []
med_ls_rmse = []
med_tls_em_rmse = []
med_ls_em_rmse = []
for j in range(n):  # 调整噪声大小
    tls_rmse = []
    ls_rmse = []
    tls_em_rmse = []
    ls_em_rmse = []
    copy_data = copy.deepcopy(data_x)

    # j=j*0.05
    times = []
    times.append(0.01* j)
    times.append(0.009 * j)
    times.append(0.06* j)

    # times=[]
    # times.append(random.randint(0, 1) * j )
    # times.append(random.randint(0, 1) * j )
    # times.append(random.randint(0, 1) * j )

    print('j:',j)
    print('times[]=:', times)

    for p in range(s): # 分割数据
        # 划分训练集与测试集
        np.random.seed(4)
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


        # data_train_random = random_datax.iloc[:N_train, [0,1,2,3,4,5,6,7]]
        # #print('data_train_random',data_train_random)
        # data_train_random['cyclelife']=np.log10(np.array(data_train_random['cyclelife']))
        # # Y_train_random = np.log10(np.array(random_datax.iloc[:N_train, 5])).reshape(-1, 1)
        # # print('data_train_random',data_train_random['cyclelife'])
        # # print('Y_train_random', Y_train_random)
        # X_test_random = random_datax.iloc[N_train:, 0:5]
        # Y_test_random = np.log10(np.array(X_test_random.iloc[N_train:, 5])).reshape(-1, 1)
        # print(data_train_random)


        for k in range(m):  # 生成m次噪声

            X_train = copy.deepcopy(data_train_random)
            #Y_train = copy.deepcopy(data_train_random.iloc)
            X_test_random=X_test_random.iloc[:,0:5]
            X_test = copy.deepcopy(X_test_random)#dataframe
            Y_test = copy.deepcopy(Y_test_random)
            # print('X_test',X_test)
            # print('Y_test',Y_test)
            standard_X = np.std(X_train, axis=0)#.reshape(-1, 1)
            #standard_Y = np.std(Y_train, axis=0)#.reshape(-1, 1)

            length0 = len(X_train[X_train['class' ]== 0])
            length1 = len(X_train[X_train['class' ]== 1])
            length2 = len(X_train[X_train['class' ]== 2])
            noise_X0 = np.random.normal(loc=0, scale=times[0], size=(length0, 6))
            noise_X1 = np.random.normal(loc=0, scale=times[1], size=(length1, 6))
            noise_X2 = np.random.normal(loc=0, scale=times[2], size=(length2, 6))

            noise_X=np.concatenate((np.concatenate((noise_X0,noise_X1),axis=0),noise_X2),axis=0)
            # print('noise_X is :',noise_X)


            X_train_noise = copy.deepcopy(X_train).values
            # Y_train_noise=np.array(X_train_noise[:,5]).reshape(-1,1)
            # print('X_train_noise',X_train_noise)
            # Y_train_noise = copy.deepcopy(Y_train)

            for index in range(len(X_train_noise)):
                flag=int(X_train_noise[index][6])
                # print('flag',flag)
                for i in range(6):
                    noise_X[:, i] *= (standard_X[i])  # 根据每个特征的标准差生成噪声
                    X_train_noise[:, i] += noise_X[:, i]


            # 转换数据类型（前面使用DataFrame是因为之前要进行特征选择）
            x_train = X_train_noise
            Y_train_noise=np.array(X_train_noise[:,5]).reshape(-1,1)
            x_test = X_test.values


            # print('X_train:',X_train)#dataframe,112*8
            # print('Y_train_noise:', type(Y_train_noise))#array
            # print('x_train:', x_train)#array,112*8




            # print('x_train is :',x_train)
            # print('type x_train is:',type(x_train))
            # print('Y_train_noise is :', Y_train_noise)
            # print('type Y_train_noise is:', type(Y_train_noise))
            # print('x_test is :', x_test)
            # print('type x_test is:', type(x_test))
            # print('Y_test is :', Y_test)
            # print('type Y_test is:', type(Y_test))

            # 总体最小二乘
            W_tls, b_tls, = tls(x_train[:,0:5], Y_train_noise)#x:array,y:array,(-1,1)
            W_tls_em,b_tls_em=add_em(x_train, Y_train_noise,'tls',x_test,Y_test)#x:array,y:array,(-1,1)
            y_pred_tls = np.dot(x_test, W_tls) + b_tls
            y_pred_tls_em = np.dot(x_test, W_tls_em) + b_tls_em
            tls_rmse.append(rmse(Y_test, y_pred_tls))
            tls_em_rmse.append(rmse(Y_test, y_pred_tls_em))

            # 最小二乘
            W_ls, b_ls, = ls(x_train[:,0:5], Y_train_noise)
            W_ls_em, b_ls_em = add_em(x_train, Y_train_noise, 'ls',x_test,Y_test)
            y_pred_ls = np.dot(x_test, W_ls) + b_ls
            y_pred_ls_em = np.dot(x_test, W_ls_em) + b_ls_em
            ls_rmse.append(rmse(Y_test, y_pred_ls))
            ls_em_rmse.append(rmse(Y_test, y_pred_ls_em))



        print('tls_rmse:',tls_rmse)
        print('tls_em_rmse:', tls_em_rmse)

    #med_tls_rmse.append(np.median(tls_rmse))
    med_ls_em_rmse.append(np.median(ls_em_rmse))
    med_ls_rmse.append(np.median(ls_rmse))
    med_tls_em_rmse.append(np.median(tls_em_rmse))
    med_tls_rmse.append(np.median(tls_rmse))


# 画图tls
plt.plot(med_tls_em_rmse)
plt.plot(med_tls_rmse)

plt.legend([ 'TLS_EM','TLS']) #
plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.show()

# 画图ls
plt.plot(med_ls_em_rmse)
plt.plot(med_ls_rmse)
plt.legend(['LS_EM', 'LS']) #
plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.show()

# # 画图tls和ls
# plt.plot(med_tls_rmse)
# plt.plot(med_ls_rmse)
# plt.legend(['TLS', 'LS'])  #
# plt.xlabel('Noise level {}'.format(n*s*m))
# plt.ylabel('RMSE')
# plt.show()

