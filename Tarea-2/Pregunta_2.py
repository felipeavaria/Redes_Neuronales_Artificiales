## parte 1

import scipy.io as sio
import numpy as np


train_data = sio.loadmat('datos_p_2/train_32x32.mat')
test_data = sio.loadmat('datos_p_2/test_32x32.mat')
X_train = train_data['X'].T
y_train = train_data['y'] - 1
X_test = test_data['X'].T
y_test = test_data['y'] - 1
#se cae por fata de memoria
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
n_classes = len(np.unique(y_train))
print 'numero de clases'
print  n_classes 
print 'clases '
clases = np.unique(y_train)
print clases
for i in clases :
	print 'clase '
	print i
	a = 0
	for data in y_train :
		
		if data == i :
			print 'lugar'
			# aca hay que mostrar la imagen ejemplo de la clase 
			print X_train[a]
			print a
			break
		a = a + 1

#mostar imagenes de entrenamiento  y de prueba
i=0
while i<5 :
	#aca hay que mostrar imagenes de ejemplo
	print X_train[i]
	i = i + 1
i=0
while i<5 :
	#aca hay que mostrar imagenes de ejemplo
	print X_test[i]
	i = i + 1


## parte 2

from keras.utils import np_utils	
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

## parte 3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu',input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(512, 7, 7, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.summary()
adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adagrad, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=1280, nb_epoch=12, verbose=1, validation_data=(X_test, Y_test))

