import numpy as np
import pandas as pd
from data_preprocessing import *
import matplotlib.pyplot as plt

"""
Once we have load the data we visualize and apply PCA


"""
plt.figure(figsize=(10,10))
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
            
           
            
