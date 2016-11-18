import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.1
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
from keras.preprocessing.image import ImageDataGenerator
#categorical_crossentropy

DIR = sys.argv[1]

all_label = pickle.load(open(DIR+'all_label.p','rb'))
unlabel = pickle.load(open(DIR+'all_unlabel.p','rb'))
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

x_unlabel = np.array(unlabel)
x_unlabel = x_unlabel.reshape(x_unlabel.shape[0], 3, 32, 32).astype('float32')
x_unlabel = x_unlabel/255


elu = ELU(alpha=1.0)
model = Sequential()
model.add(Convolution2D(64, 3 ,3, border_mode='same',dim_ordering='th', input_shape=(3, 32, 32)))
model.add(elu)
model.add(Convolution2D(64, 3 ,3 ,border_mode='same',dim_ordering='th'))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D( (3,3), strides=(2,2),dim_ordering='th') )
model.add(Dropout(0.25))
model.add(Convolution2D(128, 3 ,3 ,border_mode='same',dim_ordering='th'))
model.add(elu)
model.add(Convolution2D(128, 3 ,3 ,border_mode='same',dim_ordering='th'))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D( (3,3), strides=(2,2),dim_ordering='th') )
model.add(Dropout(0.25))
model.add(Convolution2D(256, 3 ,3 ,border_mode='same',dim_ordering='th'))
model.add(elu)
model.add(Convolution2D(256, 3 ,3 ,border_mode='same',dim_ordering='th'))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D( (3,3), dim_ordering='th') )
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500))
model.add(elu)
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
                    samples_per_epoch=len(x_train), nb_epoch=100)


flag = np.zeros(45000)
for _ in range(3):
    result = model.predict(x_unlabel,batch_size=100, verbose=0)
    index = np.argmax(result, axis=1)
    for i in range(x_unlabel.shape[0]):
        if result[i][index[i]] > 0.99 and flag[i] == 0:
            x_train = np.concatenate((x_train, np.expand_dims(x_unlabel[i], axis=0)), axis=0)
            tmp = np.zeros((1,10))
            tmp[0, np.argmax(result[i])] = 1
            y_train = np.concatenate((y_train, tmp), axis=0)
            flag[i] = 1
    print (x_train.shape, y_train.shape)
    # Data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=100),
                    samples_per_epoch=len(x_train), nb_epoch=40)
model.save(sys.argv[2])




