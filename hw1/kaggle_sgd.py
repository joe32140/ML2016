import numpy as np
import math

numberofiter = 1000

data_path = 'train.csv'
test_path = 'test_X.csv'
# Parsing Data
my_data = np.genfromtxt (data_path, delimiter=",")
my_data = my_data[1:]
my_data = my_data[:, 3:]
x = np.empty(shape=(0,162))
y = np.empty(shape=(1, 0))
#print my_data.shape
for i in range(12):
    temp = np.empty(shape=(18,0))
    for j in range(20):
        temp = np.concatenate((temp, my_data[i*360+j*18 : i*360+j*18+18 , :]) , axis = 1)

    for r in range(480):
        if math.isnan(temp[10, r]):
            temp[10, r] = 0.0

    for k in range(480-10):
        x = np.append(x, temp[:, k:k+9].reshape((1, 162)), axis = 0)
        y = np.append(y, temp[9, k+9])


test_data = np.genfromtxt (test_path, delimiter=",")
test_data = test_data[:, 2:]
test_set = np.empty(shape=(0, 162))

for i in range(240):
    for k in range(9):
        if math.isnan(test_data[i*18+10, k]):
            test_data[i*18+10, k] = 0.0
    test_set = np.append(test_set, test_data[i*18:i*18+18, :].reshape((1, 162)), axis = 0)

# Gradient descent

theta = np.random.random((162,1))
bias = np.random.random(1)
alpha = 0.00000005
converge = False
m = x.shape[0]
it = 0
min_cost = 9999999999
best_theta = theta
while not converge:
    cost = 0.0
    # Run SGD randomly
    for i in np.random.randint(m , size = m):
        # Hypothesis of answer
        hypothesis = np.dot(x[i].reshape((1, 162)), theta) + bias
        # Here is the derivation of loss function , the regularization will add in gradient step
        loss = hypothesis - y[i]
        # Sum up all the cost in an epoch
        cost += loss**2/m
        # Gradient with regularization lamda = 2
        gradient = x[i].reshape((162, 1))*loss + 4*theta
        # Update the theta and bias
        theta = theta - alpha*gradient
        bias  = bias - alpha*loss

    print("Iteration %d | Cost: %f" % (it, cost))
    if cost < min_cost:
        min_cost = cost
        best_theta = theta
    # Stop condition
    if cost <= 31:
        converge = True
    it += 1
    if it >= numberofiter:
        converge = True

print ('cost: %f', min_cost)
# Testing
ans = open('kaggle_best.csv', 'w+')
ans.write('id,value\n')
for i in range(240):
    predict = np.dot(test_set[i].reshape((1, 162)), best_theta) + bias
    ans.write('id_%d,%f\n' %(i, predict))
ans.close()



    
