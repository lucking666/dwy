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
_class = [0] * 40  + [1] * 41 + [2] * 43
data['class']=_class
_xita=[0]*124
data['xita']=_xita
print(data)
#加上噪声
def addnoise(data):
    for i in data.index:
        noise=random.gauss(0, 0.1)
        for item in items:
            data.loc[i,item]=data.loc[i,item]+noise
        data.loc[i, 'cyclelife'] = data.loc[i, 'cyclelife'] + noise
    return data


# 直接按照求解模型系数的公式得到

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

def iterate(data,X,Y, w,b):
    print("the model's parameter is :",b,w)
    Y_pred=get_y_pred(X,w,b)
    #获取y的预测值和y的真实值之间的误差
    xita=np.array(Y_pred)-Y.reshape(-1,1)
    _data=data
    _data['xita']=xita
    print('样本误差：',xita[0])
    print("0批次的方差:", math.sqrt(data[data['class']==0]['xita'].var()))

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

def rmse(y_true, y_pred):
    return np.sqrt(sum(np.square(y_true - y_pred)) / len(y_true))

def get_y_pred(x,w,b):
    # print('w',w)
    # print('b',b)
    # print('x',x)
    # print('x[i].dot(w):',x[0].dot(w))
    # print("b[0][0]:",b[0][0])
    y_pr=[]
    for i in range(len(x)):
        y_pr.append(x[i].dot(w)+b[0][0])
    return y_pr

def gettlsw_b(data):
    X = data[['F2', 'F3', 'F5', 'F6', 'F9']].values
    Y = data['cyclelife'].values
    w_tls, b_tls = tls(X, Y.reshape(-1, 1))
    print("TLS's X_b and omega is :")
    print(b_tls)
    print(w_tls)
    return w_tls,b_tls

def gettlsemw_b(data):
    TLS_EM_mae, TLS_EM_rmse = 1000, 1000
    i = 0
    w = np.ones((5, 1))
    b = [[1]]
    tls_EM=[]
    while i<200:  # ，TLS_EM_mae>TLS_mae or TLS_EM_rmse>TLS_rmse or data['xita'][0] > 1e-4
        print('i:', i)
        i = i + 1
        rowdata = data 
        X = rowdata[['F2', 'F3', 'F5', 'F6', 'F9']].values
        Y = rowdata['cyclelife'].values

        X_, Y_ = iterate(rowdata, X, Y, w, b)
        # 求解模型系数的过程，矩阵求法
        w_tlsem, b_tlsem = tls(X_, Y_.reshape(-1, 1))
        y_pr_tlsem = np.array(get_y_pred(X, w_tlsem, b_tlsem)).reshape(-1, 1)
        TLS_EM_mae, TLS_EM_rmse = geterror(y_pr_tlsem, Y)
        tls_EM.append(TLS_EM_rmse)
        # print("TLS_EM's mae is:{},TLS_EM's rmse is:{}".format(TLS_EM_mae, TLS_EM_rmse))
        # print("------------------------------------------------------------------")
        w = w_tlsem
        b = b_tlsem

    print("TLS_EM's X_b and omega is :")
    print(b_tlsem)
    print(w_tlsem)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(tls_EM)
    plt.legend(['TLS_EM'])
    plt.xlabel('data size')
    plt.ylabel('RMSE')
    # plt.xticks( range(0,n+1,5)  )
    plt.title("TLS_EM_RMSE", fontsize=14)
    plt.show()
    return w_tlsem,b_tlsem



if __name__ == '__main__':
    #首先求出tls方法得到的回归误差
    tls_list = []
    tlsem_list = []
    for i in range(1):
        rate=0.9
        train, test=splitdataset(data,rate)
        #train=addnoise(train)
        #得到tls和tlsem的模型系数
        w_tls, b_tls=gettlsw_b(train)
        w_tlsem, b_tlsem=gettlsemw_b(train)

        #测试集合上面进行测试
        x_test=test[['F2', 'F3', 'F5', 'F6', 'F9']].values
        y_test=test[['cyclelife']].values
        tls_mae,tls_rmse=geterror(get_y_pred(x_test,w_tls,b_tls),y_test)
        tlsem_mae, tlsem_rmse = geterror(get_y_pred(x_test, w_tlsem, b_tlsem), y_test)
        # tls_rmse = rmse(get_y_pred(x_test, w_tls, b_tls), y_test)
        # tlsem_rmse = rmse(get_y_pred(x_test, w_tlsem, b_tlsem), y_test)
        tls_list.append(tls_rmse)
        tlsem_list.append(tlsem_rmse)
        # print("TLS's mae is:{},TLS's rmse is:{}".format(tls_mae,tls_rmse))
        # print("TLSEM's mae is:{},TLSEM's rmse is:{}".format(tlsem_mae, tlsem_rmse))
    # print("TLS's rmse is :", np.median(tls_list))
    # print("TLS_EM's rmse is :", np.median(tlsem_list))


