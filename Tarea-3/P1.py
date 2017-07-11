# redes neuronales, tarea 3, pregunta 1 


# (a) cargar datos :

import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot
from keras.preprocessing import sequence


np.random.seed(3)
(X_train, y_train),(X_test,y_test) = imdb.load_data(seed=15)


# (b) ver cumplimento ley de zpif, al ver la representacion grafica podemos ver como es muy denso el uso de palabras de frecuencias altas (valores bajos en la indexacion), y como las palabras de baja frecuencia con valores cercanos a lo 2500 casi no ocurren
# por lo tanto si se cumple la ley de Zipf ya que un numero pequeno de palabras se repiten muchas veces y un gran numero de palabras son poco empleadas. 
'''
X = np.concatenate((X_train,X_test),axis=0 )
y = np.concatenate((y_train,y_test),axis=0 )
from matplotlib import pyplot
print("Review Length : ")
print(X)
result = map(len,X)
print(result)
pyplot.boxplot(result)
pyplot.show()
'''

# (c) podemos nota

(X_train, y_train),(X_test,y_test) = imdb.load_data(num_words=3000, seed=15) #num_words palabras mas frecuentes en este caso las 3000
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)



# se debe rellenar con ceros debido a que el cero corresponde a una palabra cualquiera que es desconcida, por ello cuando aparecen palabras no frecuentes intermedias podriamos distguir secuencias que no ocurren y que si solo se turcaran las palabras intermedias perderia sentido el comentario.


# (d) LSTM

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(3000, embedding_vector_length, input_length=500))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)


# (e) variacion de la dimension del embedding inicial (ver error de clasidicacion)
valores = []
top_words=3000
for valor in valores:
	embedding_vector_length = valor
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=500))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
	scores = model.evaluate(X_test, y_test, verbose=0)


# (f) prueba con variacion en top_words
valores = []
for valor in valores:
	embedding_vector_length = 32
	model = Sequential()
	model.add(Embedding(valor, embedding_vector_length, input_length=500))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
	scores = model.evaluate(X_test, y_test, verbose=0)



# (g) uso de dropout

from keras.layers import Dropout
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=500))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)

# (h) mejora!
















