from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import cPickle
from keras.utils import np_utils
import numpy as np
#with open('train.p', 'rb') as f:
#    data = cPickle.load(f)

data = np.load("train_sparse.npz.npy")

x_train = np.delete(data[:-1000000], 38, 1)
y_train = data[:-1000000:, 38]
y_train = np_utils.to_categorical(y_train, 5)

x_val = np.delete(data[-1000000:], 38, 1)
y_val = data[-1000000:, 38]
y_val = np_utils.to_categorical(y_val, 5)

model = Sequential()
model.add(Dense(1024, input_dim=122))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
while True:
    model.fit(x_train, y_train,
              nb_epoch=10, batch_size=128,
              validation_data=(x_val, y_val))
    model.save('model_sparse_1')
