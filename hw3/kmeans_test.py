import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.Session(config=config)
tf.python.control_flow_ops = tf
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_session(sess)
import sys
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
from keras.layers.advanced_activations import ELU
from keras.models import load_model
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib

DIR = sys.argv[1]

x_test = pickle.load(open(DIR+'test.p','rb'))
x_test = np.asarray(x_test['data'])
print (x_test.shape)
x_test = x_test.reshape(x_test.shape[0],3,32,32).astype('float32')
x_test = x_test/255

autoencoder = load_model(sys.argv[2])
get_encoder_output = K.function([autoencoder.layers[0].input],
                                  [autoencoder.layers[6].output])
hidden_test = np.zeros(shape=(0 ,1 , 128))

for i in range(x_test.shape[0]/100):
    tmp = get_encoder_output([x_test[i*100:(i+1)*100]])[0]
    hidden_test = np.concatenate((hidden_test, tmp.reshape(100, 1, 128)), axis=0)

kmeans = joblib.load(sys.argv[3]) 
result = kmeans.predict(hidden_test.reshape(hidden_test.shape[0], 128))
print (result.shape , result[0])
with open(sys.argv[4], 'w+') as f:
    f.write('ID,class\n')
    for i in range(len(result)):
        f.write('%d,%d\n' % (i, result[i]))



