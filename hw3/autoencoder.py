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
from keras.layers import Input, UpSampling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
from keras.layers.advanced_activations import ELU
from keras.models import load_model, Model
from sklearn.externals import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
#categorical_crossentropy

DIR = sys.argv[1]

all_label = pickle.load(open(DIR+'all_label.p','rb'))
x_train = []
y_train = []
for i in range(10):
    x_train.extend(all_label[i][:])
    y_tmp = np.zeros(10)
    y_tmp[i] = 1
    for j in range(500):
        y_train.append(y_tmp)

x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0],3,32,32).astype('float32')
x_train = x_train/255

y_train = np.array(y_train)

unlabel = pickle.load(open(DIR+'all_unlabel.p','rb'))
x_unlabel = np.array(unlabel)
x_unlabel = x_unlabel.reshape(x_unlabel.shape[0], 3, 32, 32).astype('float32')
x_unlabel = x_unlabel/255

x_mix = np.concatenate((x_train, x_unlabel), axis=0)
print (x_mix.shape)

inputs = Input(shape=(3, 32, 32))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='th', input_shape = (3,32,32))(inputs)
x = MaxPooling2D((2, 2), dim_ordering='th',border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(x)
x = MaxPooling2D((2, 2), dim_ordering='th',border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(x)
encoded = MaxPooling2D((2, 2), dim_ordering='th',border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', dim_ordering='th', border_mode='same')(x)

autoencoder = Model(inputs, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_mix, x_mix,
                nb_epoch=100,
                batch_size=128,
                shuffle=True)
autoencoder.save(sys.argv[2])
#autoencoder = load_model('encoder.mod')
get_encoder_output = K.function([autoencoder.layers[0].input],
                                  [autoencoder.layers[6].output])
hidden_train = np.zeros(shape=(0 ,1 , 128))
hidden_mix = np.zeros(shape=(0, 1, 128))

print ( get_encoder_output([x_train[:2]])[0].shape)
for i in range(50):
    tmp = get_encoder_output([x_train[i*100:(i+1)*100]])[0]
    hidden_train = np.concatenate((hidden_train, tmp.reshape(100, 1, 128)), axis=0)
for i in range(500):
    tmp = get_encoder_output([x_mix[i*100:(i+1)*100]])[0]
    hidden_mix = np.concatenate((hidden_mix, tmp.reshape(100, 1, 128)), axis=0)

label_hidden = np.zeros(shape=(10, 128))
for i in range(10):
    label_hidden[i] += (sum(hidden_train[i*500: (i+1)*500])/500).reshape(128,)
    
kmeans = MiniBatchKMeans(init=label_hidden, n_clusters=10, batch_size=128,
                      n_init=10, max_no_improvement=10, verbose=0)

kmeans.fit(hidden_mix.reshape(50000, 128))
joblib.dump(kmeans, sys.argv[3])

