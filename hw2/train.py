import numpy as np
import math
import sys

data_path = sys.argv[1]
model_name = sys.argv[2]
is_validation = False

# Parsing Data
my_data = np.genfromtxt (data_path, delimiter=",")

if is_validation:
  x = my_data[:3500, 1:-1]
  y = my_data[:3500, 58]
else:
  x = my_data[:, 1:-1]
  y = my_data[:, 58]

# Normalization
mean = sum(x)/x.shape[0]
sd = np.std(x, axis= 0)
x = (x - mean)/sd

# Spam select
is_spam = [int(i) for i in range(x.shape[0]) if y[i] == 1]
not_spam = [int(i) for i in range(x.shape[0]) if y[i] == 0]
bad_email = x[is_spam, :]
good_email = x[not_spam, :]

print (bad_email.shape, good_email.shape)

# Count
g_m = good_email.shape[0]
b_m = bad_email.shape[0]
count = np.zeros((1, 57))
count[0, 0] = g_m
count[0, 1] = b_m

good_mean = (sum(good_email)/g_m).reshape((1, 57))
bad_mean = (sum(bad_email)/b_m).reshape((1, 57))

# Building covariance matrix
g_sig = np.zeros((57, 57))
for i in range(g_m):
  g_sig += np.dot( (good_email[i] - good_mean).T, (good_email[i]  - good_mean) )
g_cov = g_sig/g_m

b_sig = np.zeros((57, 57))
for i in range(b_m):
  b_sig += np.dot( (bad_email[i] - bad_mean).T , (bad_email[i] - bad_mean) )
b_cov = b_sig/b_m 

share_cov = (g_cov*g_m + b_cov*b_m)/(g_m+b_m)

# Model saving 
model = np.concatenate((share_cov, good_mean), axis=0)
model = np.concatenate((model, bad_mean), axis=0)
model = np.concatenate((model, count), axis=0)
np.savetxt(model_name, model)
print (model.shape)
