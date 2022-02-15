import numpy as np
import pandas as pd

"""
Load data from MaternalDataset.csv and preprocess them

"""


#Load the data in pandas dataframe
df = pd.read_csv("./MaternalDataset.csv")
raw_data = df.values
N,M = raw_data.shape
data = np.asarray(raw_data)

print(raw_data)

#Transform the values of the last attribute to numerical
for i in range(N):
    if data[i,6]=="low risk":
        data[i,6]=1
    elif data[i,6]=="mid risk":
        data[i,6]=2
    else:
        data[i,6]=3             

#Checking for missing values
count = 0
for i in range(N):
    for j in range(M):
        if not (isinstance(data[i,j],int) or isinstance(data[i,j],float)):
            count = count + 1

print("The number of missing of corrupted values is %i." %count)
print(data)

