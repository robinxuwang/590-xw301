# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:48:38 2021

@author: XuWan
"""

#CNN = True
# CNN = False
# DFF_ANN = True
#DFF_ANN = False
folds = 5 
NKEEP=60000
batch_size=int(0.1*NKEEP)
epochs=10

#MODEL
n_bottleneck=50

from keras.datasets import mnist
from keras.datasets import fashion_mnist
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
#from keras.utils import to_categorical

from keras.callbacks import CSVLogger
csv_logger = CSVLogger('6.1_log.txt', append=True, separator=';')


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(train_fashion_img, train_fashion_labels), (test_fashion_img, test_fashion_labels) = fashion_mnist.load_data()

train_images = train_images/np.max(train_images)
train_fashion_img = train_fashion_img/np.max(train_fashion_img)

train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

train_fashion_img = train_fashion_img.reshape((60000, 28*28))
test_fashion_img = test_fashion_img.reshape((10000, 28*28))

###############################################################################
                              #DFF AE#
###############################################################################

model = models.Sequential()
model.add(layers.Dense(n_bottleneck, activation='relu', input_shape=(28*28, )))
model.add(layers.Dense(28*28, activation='relu')) 
       
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()
history = model.fit(train_images, train_images, epochs=epochs, batch_size=1000,validation_split=0.2,callbacks = [csv_logger])
history_dict = history.history


loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

results = model.evaluate(train_images, train_images)
print('The loss of train MNIST is:', results)
results_f = model.evaluate(train_fashion_img, train_fashion_img)
print('The loss of train MNIST_Fashion is:', results_f)

plt.plot(loss_values, 'r', label='Training loss')
plt.plot(val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss of MNIST')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('6.1_loss_hist.png')
plt.show()

plt.clf()

#anomalies detection for the fashion MINST DATASET
threshold = 4 * model.evaluate(train_images,train_images)
predict_result = model.predict(train_fashion_img)

#loop through the data to find any data points that is beyond the threshold
count = 0  
for i in range(train_fashion_img.shape[0]):
    error = np.mean((train_fashion_img[i] - predict_result[i])**2)  
    if threshold < error:
        count += 1
        
print('The number of anomaly in the data is: ',count )




