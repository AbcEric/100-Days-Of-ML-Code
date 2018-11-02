#
# K-NN（K Nearest Neighbors）: K近邻分类算法。
#
# 对数据分布无任何要求（non-parametric），而且是instance-based的算法（记忆而不是学习，lazy！），计算到各类别的距离。
# 距离包括Euclidean，Hamming，Manhattan，Minkowski distance；
# 需要找到合适的k值，k过小会导致噪声有较大的影响

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../datasets/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)     # 确定K值和使用的距离
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)