import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
from linear_regression_std import tls,ls
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import random
import math


data = pd.read_csv('dataset.csv')
data=data[['F2','F3','F5','F6','F9','cyclelife']]
data['cyclelife']= np.log10(data['cyclelife'])

items=['F2','F3','F5','F6','F9']
_class = [0] * 41 + [1] * 43 + [2] * 40
data['class']=_class
_xita=[10]*124
data['xita']=_xita
print(data)
#加上噪声
def addnoise(data):
    for i in data.index:
        for item in items:
            data.loc[i,item]=data.loc[i,item]+random.gauss(0, 0.1)
    return data
# print("-----------------------------------------------------------------------------------")
# data=addnoise(data)
# print(data)

#划分数据集
def splitdataset(data,rate):
    index=[[0,41],[41,84],[84,123]]
    for i in range(3):
        df=data.loc[index[i][0]:index[i][1],:]
        dftrain=df.sample(frac=rate)
        dftest=df[~df.index.isin(dftrain.index)]
        if i==0:
            train=dftrain
            test=dftest
        else:
            train=pd.concat([train,dftrain])
            test =pd.concat([test,dftest])
    return train,test

# train,test=splitdataset(data,0.5)
# print('train',train)
# train=addnoise(train)
# print('train',train)
def get_error(x,w,b):
    y_predict=[]
    for i in range(len(x)):
        y_predict.append(x[i].dot(w)+b[0][0])
    return y_predict


def iterate(data,X,Y, w,b):
    print("the model's parameter is :",b,w)
    Y_pred=get_y_pred(X,w,b)
    #获取y的预测值和y的真实值之间的误差
    xita=np.array(Y_pred)-Y.reshape(-1,1)
    _data=data
    _data['xita']=xita
    print("0批次的方差倒数:", 1/data[data['class']==0]['xita'].var())

    #更新X和Y变成与加权的X和Y
    for index in range(3):
        lamuda=1/math.sqrt(_data[_data['class']==index]['xita'].var())
        for item in items:
            _data.loc[_data['class']==index,item]=_data[_data['class']==index][item]*lamuda
        _data.loc[_data['class'] == index, 'cyclelife'] = _data[_data['class'] == index]['cyclelife'] * lamuda
    X_,Y_=_data[['F2','F3','F5','F6','F9']].values,_data['cyclelife'].values
    return X_,Y_


def geterror(y_predict,y_test):
    # print("y_pred",y_predict)
    # print("y_test",y_test)
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    return mae,rmse


def get_y_pred(x,w,b):#求最终的预测值和真实值
    # print('w',w)
    # print('b',b)
    # print('x',x)
    # print('x[i].dot(w):',x[0].dot(w))
    # print("b[0][0]:",b[0][0])
    y_pr=[]
    for i in range(len(x)):
        y_pr.append(x[i].dot(w)+b[0][0])
    return y_pr

def getlsw_b(data):
    X = data[['F2', 'F3', 'F5', 'F6', 'F9']].values
    Y = data['cyclelife'].values
    w_ls, b_ls = ls(X, Y.reshape(-1, 1))
    print("LS's X_b and omega is :")
    print(b_ls)
    print(w_ls)
    return w_ls,b_ls

def getlsemw_b(data):
    LS_EM_mae, LS_EM_rmse = 1000, 1000
    i = 0
    w = np.ones((5, 1))
    b = [[1]]
    ls_EM=[]
    while i<50:  # ，LS_EM_mae>LS_mae or LS_EM_rmse>LS_rmse or data['xita'][0] > 1e-4
        print('i:', i)
        i = i + 1
        rowdata = data
        X = rowdata[['F2', 'F3', 'F5', 'F6', 'F9']].values
        Y = rowdata['cyclelife'].values

        X_, Y_ = iterate(rowdata, X, Y, w, b)
        # 求解模型系数的过程，矩阵求法
        w_lsem, b_lsem = ls(X_, Y_.reshape(-1, 1))
        y_pr_lsem = np.array(get_y_pred(X, w_lsem, b_lsem)).reshape(-1, 1)
        LS_EM_mae, LS_EM_rmse = geterror(y_pr_lsem, Y)
        ls_EM.append(LS_EM_rmse)
        print("LS_EM's mae is:{},LS_EM's rmse is:{}".format(LS_EM_mae, LS_EM_rmse))
        print("------------------------------------------------------------------")
        w = w_lsem
        b = b_lsem
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(ls_EM)
    plt.legend(['LS_EM'])
    plt.xlabel('data size')
    plt.ylabel('RMSE')
    # plt.xticks( range(0,n+1,5)  )
    plt.title("LS_EM_RMSE", fontsize=14)
    plt.show()
    return w_lsem,b_lsem

# 对列表进行排序，针对列表长度是奇数还是偶数的不同情况，计算中位数。



if __name__ == '__main__':
    #首先求出ls方法得到的回归误差
    ls_list=[]
    lsem_list=[]
    for i in range(1):
        rate=0.9
        train, test=splitdataset(data,rate)
        # train=addnoise(train)


        #得到ls和lsem的模型系数
        w_ls, b_ls=getlsw_b(train)
        w_lsem, b_lsem=getlsemw_b(train)

        #测试集合上面进行测试
        x_test=test[['F2', 'F3', 'F5', 'F6', 'F9']].values
        y_test=test[['cyclelife']].values

        ls_mae,ls_rmse=geterror(get_y_pred(x_test,w_ls,b_ls),y_test)
        lsem_mae, lsem_rmse = geterror(get_y_pred(x_test, w_lsem, b_lsem), y_test)
        ls_list.append(ls_rmse)
        lsem_list.append(lsem_rmse)
        print("LS's mae is:{},LS's rmse is:{}".format(ls_mae,ls_rmse))
        print("LSEM's mae is:{},LSEM's rmse is:{}".format(lsem_mae, lsem_rmse))
    print("LS's rmse is :",np.median(ls_list))
    print("LS_EM's rmse is :", np.median(lsem_list))

    #求两种方法的误差
    # print("LS and LS_EM :",LS_mae-LS_EM_mae,LS_rmse-LS_EM_rmse)
    #画图
    # _Y3=X_b1.dot(omega1)
    # _Y2 =X_b.dot(omega)
    # X = np.arange(len(Y3))
    # plt.plot(X, Y3,  c='blue',marker='o',linestyle=':',  label='observe')
    # plt.plot(X, _Y2, c='red', marker='*', linestyle='-',label='LS_EM')
    # plt.plot(X, _Y3, c='green', marker='+', linestyle='--', label='LS')
    # plt.legend()
    # plt.show()
    # data.sort_values(by='class',axis=0,ascending=True,inplace=True)
    # for i in range(3):
    #     name = ['F1', 'F2']
    #     for item in name:
    #         temp = 1/data[data['class'] == i][item].var()
    #         _data.loc[data['class'] == i, item] = data[data['class'] == i][item] * temp
    # # print("更新的data：", _data)