from project1 import data_import
from collections import Counter
#from matplotlib.pylab import figure, plot, subplot, xlabel, ylabel, hist, show, scatter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
#import seaborn as sns
#import pandas as pd
#import matplotlib as plt

#LINEAR REGRESSION

X = data_import()
#Prepare the data - Input/Output
y = X[:,6]
X_cl = X[:,0:6]

#Split the data to 80% train set / 20% test set
X_train, X_test, y_train, y_test = train_test_split(X_cl, y, test_size=0.20)

#Linear Regression
model = LogisticRegression()
model.fit(X_train,y_train)
LogisticRegression()

#Calculate Logistic Regression accuracy
accuracy_LR = model.score(X_train,y_train)

#Testing
y_est = model.predict(X_test)

#Residual
residual_LR = y_est-y_test


#BASELINE CLASSIFICATION


#Find the class with the most observations
max_class = 0
for i in y:
    np.count_nonzero(y==i)
    if np.count_nonzero(y==i) > max_class:
        max_class = np.count_nonzero(y==i)
        max_class_id = i
#Calculate baseline accuracy
accuracy_baseline = max_class / len(y)


#KNN


#Euclidean Distance Function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

#KNN algorithm
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


#Calculate KNN accuracy
def accuracy_KNN(y_true, y_pred):
     accuracy_KNN = np.sum(y_true == y_pred) / len(y_true)
     return accuracy_KNN

#Run KNN
k = 4
clf = KNN(k=k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

#KNN residual
residual_KNN = predictions - y_test


#Final results
print("Logistic Regression accuracy", accuracy_LR)
print("Baseline accuracy", accuracy_baseline)
print("KNN accuracy",accuracy_KNN(y_test, predictions))