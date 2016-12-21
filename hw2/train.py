import numpy as np
import math
import sys
def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1 / (1 + np.exp(gamma))
  return 1 / (1 + np.exp(-gamma))

numberofiter = 20000
data_path = sys.argv[1]
is_validation = False


# Parsing Data
my_data = np.genfromtxt (data_path, delimiter=",")

if is_validation:
  x = my_data[:3500, 1:-1]
  y = my_data[:3500, 58]
else:
  x = my_data[:, 1:-1]
  y = my_data[:, 58]


# Gradient descent

theta = np.zeros((57,1))
bias = np.zeros(1)
alpha = 0.1
converge = False
m = x.shape[0]
it = 0
min_cost = 9999999999
ada = np.zeros((57,1))
ada_bias = np.zeros((1))
for i in range(numberofiter):
    # Hypothesis of answer
    hypothesis = np.dot(x, theta) + bias
    
    #fake loss
    loss = (1.0 / (1.0 + np.exp(-1.0*hypothesis)) ) - y.reshape((m, 1))
    
    #cost = -(y[i]*math.log(sigmoid(hypothesis))+(1-y[i])*math.log(sigmoid(1-hypothesis)))

    # Gradient 
    gradient = sum(x* loss).reshape((57,1))
    grad_bias = sum(loss)
    
    # Adagrad
    ada += gradient**2
    ada_bias += grad_bias**2 
    
    # Updating theta and bias
    theta = theta - alpha*gradient/(ada**0.5)
    bias  = bias - alpha*grad_bias/(ada_bias**0.5)

model = np.concatenate((theta, bias.reshape((1, 1)) ), axis = 0)
np.savetxt(sys.argv[2], model)


    
