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

#plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

V = Vh.T   

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title('Maternal Health Risks: PCA')
#Z = array(Z)
for c in range(3):
    # select indices belonging to class c:
    class_mask = y==c+1
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    print(Z[class_mask,i])
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()