#
# Day1：数据预处理（包括遗失数据的补充，数据编码和转换）
#

import numpy as np
import pandas as pd

dataset = pd.read_csv('../datasets/Data.csv')

# 1.Importing dataset: iloc用于数据切片，前面是行数，逗号后面是列！取列时用[]，如[:, [0,1]],否则结果会将列变为行
X = dataset.iloc[ : , :-1].values       # Matrix
Y = dataset.iloc[ : , 3].values         # Vector
print(X)

# 2.处理丢失的数据NaN：Handling the missing data: mean or median
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print(X)

# 3.编码：Encoding categorical data，例如"Yes"或"No"不能直接被模型使用
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])         # 对第一列进行编码
print(X)

# Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])   # OneHot: 属于所有可能的类别中的哪一种，用1&0表示！
X = onehotencoder.fit_transform(X).toarray()
print(X)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(Y)

# 4.将数据集分为训练和测试数据集：Splitting the datasets into training sets and Test sets，通常为80/20
# cross_validation被什么取代？
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print("X_train:")
print(X_train, X_test, Y_train, Y_test)

# 5.Feature Scaling: 通常机器学习算法采用欧几里得距离（Euclidean distance）,数量级的大小影响计算的权重，进行标准化处理！
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print("Scaling:")
print(X_train, X_test)