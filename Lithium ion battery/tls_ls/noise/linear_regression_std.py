#总体最小二乘与最小二乘法
import numpy as np
from sklearn.preprocessing import StandardScaler

def tls(X_train,Y_train):

    # 标准化X_train
    standard_X = np.std(X_train, axis=0).reshape(-1, 1)  # X加入噪声后的标准差
    stand_scaler = StandardScaler()
    std_X = stand_scaler.fit_transform(X_train)
    mean_X = stand_scaler.mean_.reshape(-1, 1)

    # 标准化Y_train
    mean_Y = np.array(np.mean(Y_train)).reshape(-1, 1)
    standard_Y = np.std(Y_train, axis=0).reshape(-1, 1)  # Y加入噪声后的标准差
    std_Y = (Y_train - mean_Y) / standard_Y

    #定义矩阵B
    B = np.vstack((np.hstack((np.dot(std_X.T, std_X), np.dot(-std_X.T, std_Y))),
                   np.hstack((np.dot(-std_Y.T, std_X), np.dot(std_Y.T, std_Y)))))

    #求B最小特征值对应的特征向量
    w,v = np.linalg.eigh(B)    #w特征值，v特征向量
    min_w_index = np.argsort(w)     #最小特征值对应的下标，argsort(w)将w中的元素从小到大排列，输出其对应的下标
    min_w_v= v[:, min_w_index[0]].reshape(-1,1)  #最小特征值对应的特征向量
    # min_w_v = v[min_w_index[0], :].reshape(-1, 1)

    #求模型参数
    n=std_X.shape[1]   #输入特征的个数
    std_W=(min_w_v[0:n]/min_w_v[n]).reshape(-1,1)
    W=np.dot(std_W, standard_Y) / standard_X

    #计算b
    _=0
    for i in range(n):
        _=_+std_W[i]*mean_X[i]*standard_Y/standard_X[i]
    b = mean_Y-_

    return W,b


def ls(X_train,Y_train):
    # 标准化X_train
    # print("Y_train's shape is :",Y_train.shape)
    standard_X = np.std(X_train, axis=0).reshape(-1, 1)  # 加入噪声后的标准差
    stand_scaler = StandardScaler()
    std_X = stand_scaler.fit_transform(X_train)
    # print("std_X's shape is :",std_X.shape)
    mean_X = stand_scaler.mean_.reshape(-1, 1)

    # 标准化Y_train
    mean_Y = np.array(np.mean(Y_train)).reshape(-1, 1)
    standard_Y = np.std(Y_train, axis=0).reshape(-1, 1)
    std_Y = (Y_train - mean_Y) / standard_Y
    # print("std_Y's shape is :", std_Y.shape)

    #求模型参数,Y=WX+b
    std_W = np.dot(np.dot(np.linalg.inv(np.dot(std_X.T, std_X)), std_X.T), std_Y)
    W=np.dot(std_W, standard_Y) / standard_X

    #计算b
    n=X_train.shape[1]
    _=0
    for i in range(n):
        _=_+std_W[i]*mean_X[i]*standard_Y/standard_X[i]
    b = mean_Y-_
    return W,b
