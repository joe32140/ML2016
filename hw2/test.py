import sys
import numpy as np

model_name = sys.argv[1]
model = np.loadtxt(model_name)
print (model.shape)
theta = model[:57]
print (theta.shape)
bias = model[57]
test_data = np.genfromtxt (sys.argv[2], delimiter=",")
test_data = test_data[:, 1:]
# Testing
ans = open(sys.argv[3], 'w+')
ans.write('id,label\n')
for i in range(test_data.shape[0]):
    predict = np.dot(test_data[i].reshape((1, 57)), theta) + bias
    print (predict.shape)
    if predict >= 0.5:
        predict = 1
    else :
        predict = 0
    ans.write('%d,%d\n' %(i+1, predict))
ans.close()
