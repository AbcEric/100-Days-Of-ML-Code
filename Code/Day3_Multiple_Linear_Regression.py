#
# Day3：多重线性回归，要求error同方差性？残差正态分布，没有多重共线性（自变量间相互无关联）！
#
# Dummy variable trap: 当多个dummy变量有相互依赖关系时，某个变量可以从其他变量推导出来，因此，如果有m个分类，使用m-1个类别即可！
#

# 1: Data Preprocessing:

import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('../datasets/50_Startups.csv')
print(dataset)
X = dataset.iloc[ : , :-1].values           # 研发费用、管理费用、市场费用和所在州
Y = dataset.iloc[ : ,  4 ].values           # 利润

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])            # 对州进行编码

onehotencoder = OneHotEncoder(categorical_features = [3])   # [3]?
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy Variable Trap: 编码变量减少一个
X = X[: , 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 3: Predicting the Test set results
y_pred = regressor.predict(X_test)
print("Predict:", X_test, y_pred, Y_test)

# X有多个变量：可视化显示有问题
import matplotlib.pyplot as plt
plt.plot(X_test , regressor.predict(X_test), color ='red')
plt.plot(X_test , Y_test, color ='blue')
plt.show()