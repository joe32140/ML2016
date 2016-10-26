import numpy as np
import math

def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1 / (1 + math.exp(gamma))
  return 1 / (1 + math.exp(-gamma))

numberofiter = 5000
data_path = 'spam_data/spam_train.csv'
test_path = 'spam_data/spam_test.csv'
is_validation = True


# Parsing Data
my_data = np.genfromtxt (data_path, delimiter=",")

if is_validation:
  x = my_data[:3500, 1:-1]
  y = my_data[:3500, 58]
else:
  x = my_data[:, 1:-1]
  y = my_data[:, 58]
test_data = np.genfromtxt (test_path, delimiter=",")

# Gradient descent

theta = np.random.random((57,1))
bias = np.random.random(1)
alpha = 0.1
converge = False
m = x.shape[0]
it = 0
min_cost = 9999999999
best_theta = theta
best_bias = bias
ada = np.ones((57, 1))
ada_bias = np.ones((1))
while not converge:
    cost = 0.0
    # Run SGD randomlyi
    gradient =  np.zeros((57, 1))
    bias = np.zeros((1))
    for i in np.random.randint(m , size = m):
        # Hypothesis of answer
        hypothesis = np.dot(x[i].reshape((1, 57)), theta) + bias
        # Here is the derivation of loss function , the regularization will add in gradient step
        #fake
        loss = sigmoid(hypothesis) - y[i]
        # Sum up all the cost in an epoch
        #cost += -(y[i]*math.log(sigmoid(hypothesis))+(1-y[i])*math.log(sigmoid(1-hypothesis)))
        cost += loss**2

        # Gradient with regularization lamda = 2
        gradient += x[i].reshape((57, 1))*loss/m # + 3*theta
        bias += loss/m

        # Update the theta and bias
    
    ada += gradient**2
    ada_bias += bias**2 
    
    theta = theta - alpha*gradient/(ada**0.5)
    bias  = bias - alpha*loss/(ada_bias**0.5)
    print("Iteration %d | Cost: %f" % (it, cost/m))
    if cost < min_cost:
        min_cost = cost
        best_theta = theta
        best_bias = bias
    # Stop condition
    if cost/m <= 0.065:
        converge = True
    it += 1
    if it >= numberofiter:
        converge = True
    if it%500 == 0:
        model = np.concatenate((theta, bias.reshape((1, 1)) ), axis = 0)
        np.savetxt('tmp_mod', model)
model = np.concatenate((best_theta, best_bias.reshape((1, 1)) ), axis = 0)
np.savetxt('logistic_model', model)


    
