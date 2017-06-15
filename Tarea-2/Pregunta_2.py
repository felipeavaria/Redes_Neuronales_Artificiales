## parte 1

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

## para correr las partes... p = 3,  para correr parte 3 asi sicesivamente...
p = 0


import matplotlib.image as mpimg
#http://ufldl.stanford.edu/housenumbers/train_32x32.mat
#http://ufldl.stanford.edu/housenumbers/test_32x32.mat

# solo caso extra http://ufldl.stanford.edu/housenumbers/extra_32x32.mat

train_data = sio.loadmat('train_32x32.mat')
test_data = sio.loadmat('test_32x32.mat')
X_train = train_data['X'].T
y_train = train_data['y'] - 1
X_test = test_data['X'].T
y_test = test_data['y'] - 1
#se cae por fata de memoria
X_train = X_train.reshape((X_train.shape[0],32,32,3))
X_test = X_test.reshape((X_test.shape[0],32,32,3))


n_classes = len(np.unique(y_train))
datos,tamX,tamY,channel = X_train.shape
clases = np.unique(y_train)
print 'dimensiones imagne :',tamX,'x',tamY
print 'numero de clases :', n_classes  
print 'clases :',clases

for i in clases :
	print 'clase '
	print i
	a = 0
	for data in y_test :
		
		if data == i :
			# aca hay que mostrar la imagen ejemplo de la clase 
			plt.title(y_test[a], fontsize = 30)
			plt.imshow(X_test[a])
			plt.show()
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
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
	
## parte 3
if p == 3:
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
	from keras.optimizers import SGD, Adadelta, Adagrad
	model = Sequential()
	model.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu',input_shape=X_test.shape[1:]))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(512, 7, 7, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(n_classes, activation='softmax'))

	model.summary()
	adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=adagrad, metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size=1280, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))



#parte 4 y parte 5

### parte 4 :prueba con distintos valores de filtros de convolucion y de pooling, se buscan los errores "accuracy"
### parte 5 :solo variar filtros convolucionales, ver los "tiempos" .

p=0
if p == 4:
	filtros_conv= (16 , 5 , 5)
	filtros_pooling = (10,10) ## <-- variar (3,3) ,(4,4) ,(5,5) , (10 ,10)
	
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
	from keras.optimizers import SGD, Adadelta, Adagrad
	model = Sequential()
	model.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu',input_shape=X_test.shape[1:]))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(512, 7, 7, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(filtros_pooling))
	model.add(Flatten())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(n_classes, activation='softmax'))

	model.summary()
	adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=adagrad, metrics=['accuracy'])

	model.fit(X_train, Y_train, batch_size=1280, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))

#parte 5
