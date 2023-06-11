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

# 加载数据
data_all = pd.read_csv('dataset.csv')
data = data_all[['F2', 'F3', 'F5', 'F6', 'F9', 'cyclelife']]

data['cyclelife']= np.log10(data['cyclelife'])
_class = [0] * 40  + [1] * 41 + [2] * 43
data['class']=_class
_xita=[0]*124
data['xita']=_xita
data[['F2noise', 'F3noise', 'F5noise', 'F6noise', 'F9noise']]=data[['F2', 'F3', 'F5', 'F6', 'F9']]
data['cyclelifenoise']=data['cyclelife']
print(data)

items=['F2noise', 'F3noise', 'F5noise', 'F6noise', 'F9noise','cyclelifenoise']
def addnoise(data):
    for i in data.index:
        noise=random.gauss(0, 0.1)
        for item in items:
            data.loc[i,item]=data.loc[i,item]+noise
    return data
data=addnoise(data)
print(data.columns)
data=data.values
# print(data[:,5])

# 数据集划分
data1 = data[:41, ]  # 第一批次
data2 = data[41:84, ]
data3 = data[84:, ]

data_all=np.concatenate((np.concatenate((data1,data2),axis=0),data3),axis=0)




data_x = copy.deepcopy(data1)  # ****选择数据集****
N_train = round(data_x.shape[0] * 0.7)  # 先切分70%
data_train = data_x[:N_train, ]
data_test = data_x[N_train:, ]

all_log_TLS_rmse = []
all_log_TLS_mape = []
all_TLS_rmse = []
all_TLS_mape = []
all_log_LS_rmse = []
all_log_LS_mape = []
all_LS_rmse = []
all_LS_mape = []

# 调整训练集的大小
for j in np.arange(2, 7.1, 0.5):
    log_TLS_rmse = []
    log_LS_rmse = []
    log_TLS_mape = []
    log_LS_mape = []
    TLS_rmse = []
    LS_rmse = []
    TLS_mape = []
    LS_mape = []
    data_train1 = copy.deepcopy(data_train) # 使用deepcopy函数创建一个与原始对象完全独立的新对象，以便在不影响原始对象的情况下进行操作
    for p in range(200):  # 前面的70%产生100组随机的排列
        np.random.seed(p)
        # 训练集
        np.random.shuffle(data_train1)  # 重新排序返回一个随机序列
        data_train_final = data_train1[:round(N_train * j / 7), :]  # ***在这调整训练集的大小***
        X_train = data_train_final[:, 8:13]
        y_train = data_train_final[:, 5].reshape(-1, 1)
        y_train_TLS=data_train_final[:, -1].reshape(-1, 1)

        # 测试集  （这里的测试集并没有进行随机排序，可以放外面或者改成随机排序后的）
        data_test1 = copy.deepcopy(data_test)
        X_test = data_test1[:, 0:5]
        y_test = data_test1[:, 5].reshape(-1, 1)  # 循环寿命对数值

        # 训练
        W, b = tls(X_train, y_train)
        W_ls, b_ls = ls(X_train, y_train_TLS)
        # 测试
        y_pred_tls = np.dot(X_test, W) + b
        y_pred_ls = np.dot(X_test, W_ls) + b_ls

        # 收集评价指标
        log_TLS_rmse.append(rmse(y_test, y_pred_tls))
        log_LS_rmse.append(rmse(y_test, y_pred_ls))
        log_TLS_mape.append(mape(y_test, y_pred_tls))
        log_LS_mape.append(mape(y_test, y_pred_ls))

        # LS与TLS预测值对数还原

        # 重新计算rmse，mape
        # TLS_rmse.append(rmse(y_test_re, Y_pred_tls_re))
        # LS_rmse.append(rmse(y_test_re, Y_pred_ls_re))
        # TLS_mape.append(mape(y_test_re, Y_pred_tls_re))
        # LS_mape.append(mape(y_test_re, Y_pred_ls_re))

    # 还原前的rmse和mape
    all_log_TLS_rmse.append(np.median(log_TLS_rmse))
    all_log_LS_rmse.append(np.median(log_LS_rmse))
    all_log_TLS_mape.append(np.median(log_TLS_mape))
    all_log_LS_mape.append(np.median(log_LS_mape))

# 保存所有评价指标
# all_metrics = np.empty(shape=(11, 8))
# all_metrics[:, 0] = all_log_TLS_rmse
# all_metrics[:, 1] = all_log_LS_rmse
# all_metrics[:, 2] = all_log_TLS_mape
# all_metrics[:, 3] = all_log_LS_mape
# all_metrics[:, 4] = all_TLS_rmse
# all_metrics[:, 5] = all_LS_rmse
# all_metrics[:, 6] = all_TLS_mape
# all_metrics[:, 7] = all_LS_mape
# df = pd.DataFrame(all_metrics)
# df.to_csv('all_metrics_data1.csv', index=False, header=False)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(all_log_TLS_rmse)
plt.plot(all_log_LS_rmse)
plt.legend(['TLS', 'LS'])
plt.xlabel('data size')
plt.ylabel('RMSE')
# plt.xticks( range(0,n+1,5)  )
plt.title("改变——数据集1_还原前RMSE", fontsize=14)
plt.show()

# plt.plot(all_log_TLS_mape)
# plt.plot(all_log_LS_mape)
# plt.legend(['TLS','LS'])
# plt.xlabel('data size')
# plt.ylabel('MAPE')
# # plt.xticks( range(0,n+1,5)  )
# plt.title("数据集3_还原前MAPE", fontsize=14)
# plt.show()

# plt.plot(all_TLS_rmse)
# plt.plot(all_LS_rmse)
# plt.legend(['TLS', 'LS'])
# plt.xlabel('data size')
# plt.ylabel('RMSE')
# # plt.xticks( range(0,n+1,5) )
# plt.title("数据集1_还原后RMSE", fontsize=14)
# plt.show()

# plt.plot(all_TLS_mape)
# plt.plot(all_LS_mape)
# plt.legend(['TLS','LS'])
# plt.xlabel('data size')
# plt.ylabel('MAPE')
# # plt.xticks( range(0,n+1,5)  )
# plt.title("数据集3_还原后MAPE", fontsize=14)
# plt.show()
