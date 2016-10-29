
import numpy as np
import math
import sys
def sigmoid(x):
  return  1.0/(1.0+np.exp(-x))


model_name = sys.argv[1]
test_data = sys.argv[2]
answer_path = sys.argv[3]

model = np.loadtxt(model_name)
print (model.shape)
share_cov = model[:57,:]
good_mean = model[57, :].reshape((1, 57))
bad_mean = model[58, :].reshape((1, 57))
g_m = model[59, 0]
b_m = model[59, 1]
print (g_m, b_m)

test_data = np.genfromtxt (test_data , delimiter=",")
test_data = test_data[:, 1:]

# Normalization
mean = sum(test_data)/test_data.shape[0]
sd = np.std(test_data, axis= 0)
test_data = (test_data - mean)/sd

print (test_data.shape[0])
# Testing
ans = open(answer_path, 'w+')
ans.write('id,label\n')
s = np.linalg.pinv(share_cov)
for i in range(test_data.shape[0]):

    # Calculating probability
    tmp=test_data[i].reshape((1, 57))
    w = np.dot((good_mean-bad_mean), s)
    wx = np.dot(w, tmp.T)

    b1 = np.dot(good_mean, s)
    b1 = np.dot(b1, good_mean.T)/2.0
    b2 = np.dot(bad_mean, s)
    b2 = np.dot(b2, bad_mean.T)/2.0
    a = wx-b1+b2+np.log(float(g_m)/b_m)
    
    predict = sigmoid(a)
    if predict >= 0.5:
        predict = 0
    else :
        predict = 1
    ans.write('%d,%d\n' %(i+1, predict))
ans.close()
