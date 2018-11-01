#
# Day4-6：Logistic回归
#
# 与线性回归的区别在于，Logistic回归的输出是离散的(discrete，采用Sigmoid函数), 而线性回归是连续输出。
#

# 1.Data Pre-Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2.Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 3.Predection
y_pred = classifier.predict(X_test)

# 4.Evaluating The Predection
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#plt.scatter(X_test , y_test, color = 'red')
#plt.scatter(X_test , y_pred, color = 'blue')
#plt.plot(X_train , regressor.predict(X_train), color ='blue')
#plt.show()