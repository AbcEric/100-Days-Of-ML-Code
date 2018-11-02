#
# Day4-6：LR（Logistic Regression）逻辑回归
#
# Logistics Regression是一个二分类，或者叫0-1分类；而Softmax Regression就是一个多分类（0-1-2-…）
# LR仍然属于线性回归的范畴，因为分界面是线性的
#
# 与Linear Regression的区别在于，Logistic回归的输出是离散的(discrete，采用Sigmoid函数), 而线性回归是连续输出。
# Logisitic函数：f(z)=1/(1+e^(-z))
# 损失函数(cost function), 梯度下降算法
# 从sklearn的model_selection中，取代cross_validation，避免警告！

# 1.Data Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
# print(X, y)

# Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2.Logistic Regression Model
from sklearn.linear_model import LogisticRegression         # 这里叫线性模型，是因为这里两类用户被一条直线划分开！如何避免？
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 3.Predection
y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)

# 4.评判预测结果：Evaluating The Predection，如何可视化？
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)               # 分别是A是B不是，已经A不是B是的合计数
print(cm)

#plt.scatter(X_test , y_test, color = 'red')
#plt.scatter(X_test , y_pred, color = 'blue')
#plt.plot(X_train , regressor.predict(X_train), color ='blue')
#plt.show()