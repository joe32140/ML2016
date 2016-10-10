import numpy as np
import math

numberofiter = 1

data_path = 'data/train.csv'
test_path = 'data/test_X.csv'
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

test_data = np.genfromtxt (test_path, delimiter=",")
test_data = test_data[:, 2:]
for i in range(240):
    for j in range(9):
        if math.isnan(test_data[i*18+10, j]):
            test_data[i*18+10, j] = float(0)
# Gradient Descend
theta = np.random.random(18*9).reshape((18*9, 1))
bias = np.random.random(1);
alpha = 0.000001
converge = False
it = 0
while not converge:
    for j in range(50):#5750
        hypothesis =np.dot(x[:, j:j+9].reshape((1, 18*9)),theta) + bias
        print theta.shape
        loss =  hypothesis - x[9, j+9] + 5*np.dot(theta.T, theta)
        print loss.shape
        print("Iteration %d | Cost: %f" % (j, loss**2))
        gradient = x[:, j:j+9].reshape((18*9, 1))*loss + 10*theta
        print gradient.shape
        #print gradient[:5]
        theta = theta - alpha*gradient
        bias = bias - alpha*loss
       # if loss**2 < 0.0001:
       #     converge = True
       #     break
    it = it + 1
    if  it >= numberofiter:
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
    #predict = np.dot(x[:, i*9:i*9+9].reshape(1, 18*9), theta) + bias    

    #print test_data[i:i+18, :]
    #print predict
    ans.write('id_%d,%f\n' %(i, predict))
ans.close()
