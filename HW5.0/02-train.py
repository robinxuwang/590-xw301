# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 22:44:45 2021

@author: XuWan
"""
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from tensorflow.keras import regularizers
from keras.callbacks import CSVLogger
import random



##################################################
#################    load data    ################
################################################## 
with open('texts.txt', 'r',encoding='utf-8') as f:
    texts = f.read().splitlines() 
with open('labels.txt', 'r',encoding='utf-8') as f:
    labels = f.read().splitlines() 

##################################################
############## text book codes begin #############
################################################## 
    
#params
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)


labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

##################################################
############## text book codes ends  #############
##################################################

#since the labels are not binary 
y_train = to_categorical(y_train,3)
y_val = to_categorical(y_val,3)

#train both a 1D CNN and a RNN model to predict the novel title (category) 
#based on the fragments from the text


# # fit and evaluate 1D CNN Model model

#DEFINE KERAS MODEL 
model = Sequential()

model.add(Embedding(10000,100,input_length = maxlen))
model.add(layers.Conv1D(filters=16, kernel_size=4, activation='relu',strides = 1, input_shape = (100,1000)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Conv1D(filters=8, kernel_size=4, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))


##################################################
############## text book codes begin #############
################################################## 


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])
history = model.fit(x_train, y_train,epochs=20,batch_size=128,validation_split=0.2, 
                    callbacks=[CSVLogger('cnn_log.txt', append=True, separator=';')])   #slight alternation to include the log file
#model.save_weights('pre_trained_glove_model.h5')
model.save('1DCNN')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy 1DCNN')
plt.legend()
plt.savefig('cnn_accuracy.png')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss 1DCNN')
plt.legend()
plt.savefig('cnn_loss.png')

#plt.show()
##################################################
############## text book codes end   #############
################################################## 


from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.add(Dense(3,activation='softmax'))
model.summary()


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,epochs=10,batch_size=128,validation_split=0.2,
                    callbacks=[CSVLogger('rnn_log.txt', append=True, separator=';')])

model.save('SimpleRNN')


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy simple RNN')
plt.legend()
plt.savefig('simpleRNN_accuracy.png')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss-Simple RNN')
plt.legend()
plt.savefig('simpleRNN_loss.png')
plt.show()


