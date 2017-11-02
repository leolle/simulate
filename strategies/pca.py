# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:13:26 2017

@author: gft
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

##step1:pca simple version
x=np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y=np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])
mean_x=np.mean(x)
mean_y=np.mean(y)
scaled_x=x-mean_x
scaled_y=y-mean_y

data=np.matrix([[scaled_x[i],scaled_y[i]] for i in range(len(scaled_x))])

##求协方差矩阵
cov=np.cov(scaled_x,scaled_y)
##求协方差矩阵的特征根和特征向量
eig_val,eig_vec=np.linalg.eig(cov)###eig_val是特征值,eig_vec是特征向量
##特征向量是k*1的列向量(k是feature的个数)
##eig_vec[0]是特征向量1
##eig_vec[1]是特征向量2

##aa=eig_vec[1] * eig_vec[0]
##aa[0]+aa[1]=0
##因此，特征向量之间是正交的

'''
plt.plot(scaled_x,scaled_y,'o',)
xmin,xmax=scaled_x.min(),scaled_x.max()
ymin,ymax=scaled_y.min(),scaled_y.max()
dx=(xmax-xmin)*0.2
dy=(ymax-ymin)*0.2
plt.xlim(xmin-dx,xmax+dx)
plt.ylim(ymin-dy,ymax+dy)
plt.plot([eig_vec[:,0][0],0],[eig_vec[:,0][1],0],color='red')
plt.plot([eig_vec[:,1][0],0],[eig_vec[:,1][1],0],color='red')
'''

eig_pairs=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
eig_pairs.sort(reverse=True)##选择特征值最大的特征向量
feature=eig_pairs[0][1]
##转化得到降维的数据
new_data_reduced=np.transpose(np.dot(feature,np.transpose(data)))


##step2:pca python 实现
import numpy as np
def pca(X,k):
    #k is the components you want
    n_samples, n_features = X.shape#mean of each feature
    mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
    norm_X=X - mean#normalization，每个变量都减去自己的均值
    scatter_matrix=np.dot(np.transpose(norm_X),norm_X)#scatter matrix
    #Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    #get new data
    data=np.dot(norm_X,np.transpose(feature))
    return data

##step3:sklearn PCA
from sklearn.decomposition import PCA
X=np.array([[-1,1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
pca_result=PCA(n_components=1)
pca_result.fit(X)
test=pca_result.transform(X)

##compare with:用自己定义的pca和上面的pca做对比
test1=pca(X,1)

####step4:pca python实现

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
%matplotlib inline

#Load data set
data = pd.read_csv('Big_Mart_PCA.csv')
#convert it to numpy arrays
X=data.values
#Scaling the values
X = scale(X)
pca = PCA(n_components=44)
pca.fit(X)
#The amount of variance that each PC explains
var= pca.explained_variance_ratio_
var1= pca.explained_variance_
#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print var1

#Looking at above plot I'm taking 30 variables
pca = PCA(n_components=30)
##或者pca=PCA(n_components=0.95)
#pca.n_components_
#
pca.fit(X)
X1=pca.fit_transform(X)
print X1



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1],
[2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2],
random_state =9)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')##三维数据的分布图

##不降维，只对数据进行投影，
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
print pca.explained_variance_ratio_
print pca.explained_variance_
##降维之后，从三维降到二维
pca = PCA(n_components=2)
pca.fit(X)
print pca.explained_variance_ratio_
print pca.explained_variance_
#为了有个直观的认识，我们看看此时转化后的数据分布，代码如下
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
plt.show()

#现在我们看看不直接指定降维的维度，而指定降维后的主成分方差和比例
pca = PCA(n_components=0.95)
pca.fit(X)
print pca.explained_variance_ratio_
print pca.explained_variance_
print pca.n_components_
#最后我们看看让MLE算法自己选择降维维度的效果，代码如下：
pca = PCA(n_components='mle')
pca.fit(X)
print pca.explained_variance_ratio_
print pca.explained_variance_
print pca.n_components_
