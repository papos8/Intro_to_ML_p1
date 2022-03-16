from project1 import *

import seaborn as sns
import pandas as pd
import matplotlib as plt
from matplotlib.pylab import figure, plot, subplot, xlabel, ylabel, hist, show, scatter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


#LINEAR REGRESSION


#Prepare the data - Input/Output
y = X[:,6]
X_LR = X[:,0:6]

#Split the data to 80% train set / 20% test set
X_train, X_test, y_train, y_test = train_test_split(X_LR, y, test_size=0.20)

#Linear Regression
model = LogisticRegression()
model.fit(X_train,y_train)
LogisticRegression()

#Accuracy
accuracy = model.score(X_train,y_train)

#Testing
y_est = model.predict(X_test)

#Residual
residual = y_est-y_test

#BASELINE CLASSIFICATION


#Find the class with the most observations
max_class = 0
for i in y:
    np.count_nonzero(y==i)
    if np.count_nonzero(y==i) > max_class:
        max_class = np.count_nonzero(y==i)
        max_class_id = i

baseline_accuracy = max_class / len(y)

scatter(X[:,5],X[:,4],X[:,3],X[:,1],c=y)

