#
# Day2：Simple Linear Regression简单线性回归，Y为dependent variable，X为independent variable
#

# 1.Data Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../datasets/studentscores.csv')
print(dataset)
X = dataset.iloc[ : ,   : 1 ].values        # 注意采用":"方式为前闭后开，等于0:1或[0]
Y = dataset.iloc[ : , 1 ].values            # 第1列变为行
print(X, Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

# 2.Fitting Simple Linear Regression Model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 3.Predecting the Result
Y_pred = regressor.predict(X_test)
print(Y_pred, Y_test)

# 4.Visualising the Training and Test results:
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()

plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')

plt.show()