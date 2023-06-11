import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
from linear_regression_std import tls,ls
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import random



#
# data = pd.read_csv('dataset.csv')
# data=data[['F2','F3','F5','F7','F9','cyclelife']]
# data['cyclelife']= data['cyclelife'].apply(np.log1p)
# items=['F2','F3','F5','F7','F9']
# _class = [0] * 41 + [1] * 43 + [2] * 40
# data['class']=_class
# _xita=[10]*124
# data['xita']=_xita
#
# def splitdataset(data,rate):
#     index=[[0,40],[41,83],[84,123]]
#     for i in range(3):
#         df=data.loc[index[i][0]:index[i][1],:]
#         dftrain=df.sample(frac=rate)
#         dftest=df[~df.index.isin(dftrain.index)]
#         if i==0:
#             train=dftrain
#             test=dftest
#         else:
#             train=pd.concat([train,dftrain])
#             test =pd.concat([test,dftest])
#     return train,test
#
# train,test=splitdataset(data,0.5)
# print(train)
# print(test)
#
# for it in train.index:
#     print(type(it))
#     print(it)
# data2=data.loc[84:,:]
# data21=data2.sample(frac=0.8)
# data22=data2[~data2.index.isin(data21.index)]
# print(data21)
# print(data22)
# data1t=pd.concat([data21,data22])
# print(data1t)
# if data['xita'][0]<1000:
#     print("here ")
# else :
#     print("what")
# print(1e-3)

#
# items=['F2','F3','F5','F7','F9']
# print(data.loc[0])
# print(data.loc[1])
# for i in range(len(data)):
#     noise = random.gauss(0, 1)
#     for item in items:
#         data.loc[i, item] = data.loc[i, item] + noise
#
# print(data.loc[0])
# print(data.loc[1])

# d = {'one' : pd.Series([10, 100, 1000,10000]),
#    'two' : pd.Series([10, 10000, 1000, 100])}
#
# df = pd.DataFrame(d)
# print(df)
# print(len(df[df['one']==10]))

x=np.random.normal(0,1,size=(3,6))
print(type(x))
# df=df.values
# if df[:,1]==10:
#    print(df[:,1])
# print(df)
#
# a=np.random.randint(1,100,(3,4))
# print(a)
# for index in range(len(a)):
#     flag=a[index][3]
#     print(flag)
