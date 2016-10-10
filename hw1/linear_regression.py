import numpy as np
import math

numberofiter = 8

data_path = 'data/train.csv'

# Parsing Data
my_data = np.genfromtxt (data_path, delimiter=",")
my_data = my_data[1:]
my_data = my_data[:, 3:]
x = np.empty(shape=(18,0))
for i in range(240):
    x = np.concatenate( (x, my_data[i*18:i*18+18,:]), axis=1)

for i in range(5760):
    if math.isnan(x[10, i]):
        x[10, i] = float(0)
#print x[9, :]

# Gradient Descend
theta = np.ones(18*9).reshape((18, 9))
bias = 1.0
alpha = 0.0001
print x[:, :9].shape
for i in range(numberofiter):
    for j in range(5750):#5751
        hypothesis =sum(sum(x[:, j:j+9]*theta)) + bias
        #print x[:, :9]
        loss =  hypothesis - x[9, j+10]
        print("Iteration %d | Cost: %f" % (j, loss))
        gradient = theta*loss
        theta = theta - alpha*gradient
        bias = bias - alpha*loss
