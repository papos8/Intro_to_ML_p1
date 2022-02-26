import numpy as np
import pandas as pd
from data_preprocessing import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

"""
Once we have load the data we visualize and apply PCA


"""
plt.figure(figsize=(40,40))
k=1
for i in range(7):    
    for j in range(7):
        plt.subplot(7,7,k)
        for c in range(1,len(classes)+1):
                # select indices belonging to class c:
                class_mask = y==c                   
                plt.plot(data[class_mask,i], data[class_mask,j], 'o',alpha=1)
                plt.legend(classNames)
                plt.xlabel(attributeNames[i])
                plt.ylabel(attributeNames[j])
        k=k+1   
           
#Create X matrix from raw_data
X = np.empty((1014, 7))
for i, col_id in enumerate(range(1,8)):
    X[:, i] = np.asarray(raw_data[:,col_id-1])

#Substract the mean / SVD
Y = X - np.ones((N,1))*X.mean(axis=0)     
U,S,V = svd(Y,full_matrices=False)          
rho = (S*S) / (S*S).sum() 
threshold = 0.9

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)