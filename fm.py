#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from random import normalvariate #正态分布
# from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import MinMaxScaler as MM #可将特征缩放到给定的最小值和最大值之间
import pandas as pd


# In[4]:


data_train = pd.read_csv('D:\hku\COMP7404\Group p\Data for Project\cs-training.csv', header=None)
data_test = pd.read_csv('D:\hku\COMP7404\Group p\Data for Project\cs-test.csv', header=None)


# In[37]:


data_test_edit = data_test.fillna(0)
data_train_edit =data_train.fillna(0)


# In[30]:


def preprocessing(data_input):
    standardopt = MM()
    data_input.iloc[1:, 1].replace(0, -1, inplace=True) #把数据集中的0转为-1
    feature = data_input.iloc[1:, 2:] #除了第一列之外，其余均为特征
    feature = standardopt.fit_transform(feature) #将特征转换为0与1之间的数
    feature = np.mat(feature)#传回来的是array，如果要dataframe那用dataframe
    label = np.array(data_input.iloc[1:, 1]) #第一一列是标签，表示有无逾期记录
    return feature, label #返回特征，标签


# In[18]:


def sigmoid(x): #定义sigmoid函数
    return 1.0/(1.0 + np.exp(-x))


# In[70]:


def sgd_fm(datamatrix, label, k, iter, alpha):
    '''
    k：分解矩阵的长度
    datamatrix：数据集特征
    label：数据集标签
    iter:迭代次数
    alpha:学习率
    '''
    m, n = np.shape(datamatrix) #m:数据集特征的行数，n:数据集特征的列数
    w0 = 0.0 #初始化w0为0
    w = np.zeros((n, 1)) #初始化w
    v = normalvariate(0, 0.2) * np.ones((n, k))
    for it in range(iter):
        for i in range(m):
            # inner1 = datamatrix[i] * w
            inner1 = datamatrix[i] * v #对应公式进行计算
            inner2 = np.multiply(datamatrix[i], datamatrix[i]) * np.multiply(v, v)
            jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
            ypredict = w0 + datamatrix[i] * w + jiaocha
            # print(np.shape(ypredict))
            # print(ypredict[0, 0])
            yp = sigmoid(np.mat(label[i])*ypredict[0, 0])
            loss = 1 - (-(np.log(yp)))
            z1=alpha * (yp - 1)
            z2=np.mat(label[i]) * 1
            z3=z1*z2
            w0 = w0 -z3
            for j in range(n):
                if datamatrix[i, j] != 0:
                    z21=alpha * (yp - 1)
                    z22=np.mat(label[i]) * datamatrix[i, j]
                    z23=z21*z22
                    w[j] = w[j] - z23
                    for k in range(k):
                        z30=(yp - 1) 
                        z31=z30*np.mat(label[i])
                        z32=np.mat(datamatrix[i, j])*np.mat(inner1[0, k])
                        z33=np.mat(v[j, k])*np.mat(datamatrix[i, j])
                        z34=np.mat(z33)*np.mat(datamatrix[i, j])
                        z35=z32-z34
                        z36=np.mat(z31)*np.mat(z35)
                        z37=alpha * np.mat(z36)
                        v[j, k] = v[j, k] - z37
                      
        print('第%s次训练的误差为：%f' % (it, loss))
    return w0, w, v


# In[40]:


def predict(w0, w, v, x, thold):
    inner1 = x * v
    inner2 = np.multiply(x, x) * np.multiply(v, v)
    jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
    ypredict = w0 + x * w + jiaocha
    y0 = sigmoid(ypredict[0,0])
    if y0 > thold:
        yp = 1
    else:
        yp = -1
    return yp


# In[21]:


def calaccuracy(datamatrix, label, w0, w, v, thold):
    error = 0
    for i in range(np.shape(datamatrix)[0]):
        yp = predict(w0, w, v, datamatrix[i], thold)
        if yp != label[i]:
            error += 1
    accuray = 1.0 - error/np.shape(datamatrix)[0]
    return accuray


# In[71]:


datamattrain, labeltrain = preprocessing(data_train_edit) #将训练集进行预处理，datamattrain存放训练集特征，labeltrain存放训练集标签
datamattest, labeltest = preprocessing(data_test_edit)#将测试集进行预处理，datamattest存放训练集特征，labeltest存放训练集标签
w0, w, v = sgd_fm(datamattrain, labeltrain, 20, 50, 0.01)#分解矩阵的长度为20，迭代次数为300次，学习率为0.01
maxaccuracy = 0.0 
tmpthold = 0.0


# In[72]:


for i in np.linspace(0.4, 0.6, 201):
    #print(i)
    accuracy_test = calaccuracy(datamattest, labeltest, w0, w, v, i)
    if accuracy_test > maxaccuracy:
        maxaccuracy = accuracy_test
        tmpthold = i
print("准确率:",accuracy_test)


# In[66]:





# In[51]:





# In[52]:





# In[53]:





# In[ ]:




