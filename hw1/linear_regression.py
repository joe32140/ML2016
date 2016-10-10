import numpy as np
import math

numberofiter = 2000

data_path = 'data/train.csv'
test_data = 'data/test_X.csv'
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
x_scale = (sum(x.T)/5760).reshape((18, 1));
print x_scale.shape
print x_scale
#print x[9, :]

test_data = np.genfromtxt (test_data, delimiter=",")
test_data = test_data[:, 2:]
for i in range(240):
    for j in range(9):
        if math.isnan(test_data[i*18+10, j]):
            test_data[i*18+10, j] = float(0)
#print test_data[10]
#print x[9, :]
# Gradient Descend
theta = np.zeros(18*9).reshape((18, 9))
bias = 0.0
alpha = 0.000002
for i in range(numberofiter):
    for j in range(5750):#5750
        hypothesis =sum(sum(x[:, j:j+9]*theta)) + bias
        #print x[:, :9]
        loss =  hypothesis - x[9, j+9]
        #print("Iteration %d | Cost: %f" % (j, loss))
        gradient = x[:, j:j+9]*loss
        theta = theta - alpha*gradient/x_scale
        bias = bias - alpha*loss

# Testing
print theta
print bias
ans = open('ans.csv', 'w+')
ans.write('id,value\n')
for i in range(240):
    predict = sum(sum(test_data[i:i+18, :]*theta)) + bias
    #print test_data[i:i+18, :]
    #print predict
    ans.write('id_%d,%f\n' %(i, predict))
ans.close()
