import numpy as np
import math

numberofiter = 1

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
#x_scale = (sum(x.T)/5760).reshape((18, 1));
#x = (x.T/x_scale.T).T
#print x_scale.shape
#print x_scale
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
theta = np.zeros(18*9).reshape((18*9, 1))
bias = 0.0
alpha = 0.000001
last_loss = 0.0
converge = False
it = 0
while not converge:
    for j in range(500):#5750
        hypothesis =np.dot(x[:, j:j+9].reshape((1, 18*9)),theta) + bias
        #print x[:, :9]
        print hypothesis.shape
        loss =  hypothesis - x[9, j+9]
        print("Iteration %d | Cost: %f" % (j, loss))
        gradient = x[:, j:j+9].reshape((18*9, 1))*loss
        #print gradient
        theta = theta - alpha*gradient
        bias = bias - alpha*loss
    it = it + 1
    if loss**2 < 5 or it >= numberofiter:
        converge = True  

# Testing
#print theta
#print bias
ans = open('ans.csv', 'w+')
ans.write('id,value\n')
for i in range(240):
    #query = (test_data[i:i+18, :].T/x_scale.T).T
    #predict = sum(sum(query*theta)) + bias
    predict = np.dot(test_data[i:i+18, :].reshape(1, 18*9), theta) + bias
    #print test_data[i:i+18, :]
    #print predict
    ans.write('id_%d,%f\n' %(i, predict))
ans.close()
