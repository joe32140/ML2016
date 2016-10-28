import numpy as np
import math

data_path = 'spam_data/spam_train.csv'
is_validation = False

# Parsing Data
my_data = np.genfromtxt (data_path, delimiter=",")

if is_validation:
  x = my_data[:3500, 1:-1]
  y = my_data[:3500, 58]
else:
  x = my_data[:, 1:-1]
  y = my_data[:, 58]


is_spam = [a for a in y if a == 1]
not_spam = [b for b in y if b == 0]
bad_email = x[is_spam, :]
good_email = x[not_spam, :]

g_m = good_email.shape[0]
b_m = bad_email.shape[0]
count = np.zeros((1, 57))
count[0, 0] = g_m
count[0, 1] = b_m

good_a = good_email - np.dot( np.ones((g_m, g_m)), good_email)/g_m
good_cov = np.dot(good_a.T, good_a)
good_mean = (sum(good_email)/g_m).reshape((1, 57))


bad_a = bad_email - np.dot( np.ones((b_m ,b_m)), bad_email)/b_m
bad_cov = np.dot(bad_a.T, bad_a)
bad_mean = (sum(bad_email)/b_m).reshape((1, 57))


share_cov = (good_cov*g_m + bad_mean*b_m)/(g_m+b_m)
model = np.concatenate((share_cov, good_mean), axis=0)
model = np.concatenate((model, bad_mean), axis=0)
model = np.concatenate((model, count), axis=0)
np.savetxt('gaussian_model', model)
print (model.shape)
