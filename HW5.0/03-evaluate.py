# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 22:44:46 2021

@author: XuWan
"""


import numpy as np
import keras
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
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN


##################################################
#################    load data    ################
################################################## 
with open('texts.txt', 'r',encoding='utf-8') as f:
    texts = f.read().splitlines() 
with open('labels.txt', 'r',encoding='utf-8') as f:
    labels = f.read().splitlines() 

    
#params
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)


word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)


labels = np.asarray(labels)
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
labels = to_categorical(labels,3)

##################################################
#################    eval         ################
################################################## 

model = keras.models.load_model('1DCNN')
print('The loss and accuracy of 1D CNN model are:', model.evaluate(data,labels))

model1 = keras.models.load_model('SimpleRNN')
print('The loss and accuracy of simple RNN model are: ',model1.evaluate(data,labels))









