#This file contains the summary of statistics for our dataset
import math
from data_preprocessing import *
from scipy import stats

mean = []

#Transpose data to easier access the values
t_data = data.T
print(t_data)

K,L=t_data.shape


#Create an array for the mean of the attributes
mean = []
for i in range(K):
    mean.append(round(t_data[i].mean(),2))
print("The array of the mean values is: " , mean)

#Create an array for the variance of the attributes 
var = []
for i in range(K):
    var.append(round(t_data[i].var(),2))
print("The array of the variance is: " , var)

#Create an array for the svd of the attributes
std = []
for i in range(K):
    std.append(round(math.sqrt(var[i]),2))
print("The array of the standard deviation is: " , std)

#Create an array with the range of the values
v_range = []
for i in range(K):
    v_range.append(t_data[i].max()-t_data[i].min())
print("The array of the range is: " , v_range)

#Determine mix and max
v_min = []
v_max = []
for i in range(K):
    v_min.append(t_data[i].min())
    v_max.append(t_data[i].max())
print("The array of the min values is: " , v_min)
print("The array of the max values is: " , v_max)

#Compute mode of last attribute
mode = stats.mode(t_data[6])
#Print the mode (1)
print("The mode of the risk level is", mode[0][0])
