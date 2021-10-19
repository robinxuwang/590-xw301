# -*- coding: utf-8 -*-

#------------------
#Flags
#------------------
#CNN = True
CNN = False
DFF_ANN = True
#DFF_ANN = False
folds = 5 
NKEEP=60000
batch_size=int(0.1*NKEEP)
epochs=20

#------------------
#data
#------------------
data = {"mnist":0,"mnist_fashion":1,"cifar10":2}
default = "mnist"

#MODIFIED FROM CHOLLETT P120 
from keras import layers 
from keras import models
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.summary()
#-------------------------------------
#GET DATA AND REFORMAT
#-------------------------------------
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

#NORMALIZE
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255  

#DEBUGGING
print("batch_size",batch_size)
rand_indices = np.random.permutation(train_images.shape[0])
train_images=train_images[rand_indices[0:NKEEP],:,:]
train_labels=train_labels[rand_indices[0:NKEEP]]
# exit()


#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)


def vis(data,sample):
    img = data[sample]
    plt.figure(figsize=(2,2))
    plt.axis('off')
    plt.imshow(img,interpolation = 'nearest')

dim = [train_images.shape[0],train_images.shape[1],train_images.shape[2],train_images.shape[3]]

#-------------------------------------
#BUILD MODEL CNN or DFF_ANN
#-------------------------------------
acc_per_fold = []
loss_per_fold = [] 

if CNN == True:
    kfold = KFold(n_splits = folds, shuffle = True)
    fold_count = 1
    for train, test in kfold.split(train_images,train_labels):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(dim[1], dim[2], dim[3])))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128,(3,3),activation = "relu"))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        history = model.fit(train_images[train], train_labels[train], epochs = epochs, batch_size = batch_size)
        scores = model.evaluate(train_images[test], train_labels[test], batch_size = train_images[test].shape[0]) 
        print(f'Score for fold {fold_count}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_count += 1 
    print("average accuracy = ", np.mean(acc_per_fold))
    print("average loss = ", np.mean(loss_per_fold))
if DFF_ANN == True:
    train_images = train_images.reshape((dim[0], dim[1] * dim[2] * dim[3]))
    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2] * test_images.shape[3]))
    kfold = KFold(n_splits=folds, shuffle=True)
    fold_count = 1
    for train, test in kfold.split(train_images, train_labels):
        model = models.Sequential()
        #ADD LAYERS
        model.add(layers.Dense(512, activation='relu', input_shape=(dim[1] * dim[2] * dim[3],)))
        #SOFTMAX  --> 10 probability scores (summing to 1
        model.add(layers.Dense(10,  activation='softmax'))
        #COMPILATION (i.e. choose optimizer, loss, and metrics to monitor)
        model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        history = model.fit(train_images[train], train_labels[train], epochs = epochs, batch_size = batch_size)
        scores = model.evaluate(train_images[test], train_labels[test], batch_size = train_images[test].shape[0])
        print(f'Score for fold {fold_count}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_count += 1
    print("average accuracy = ", np.mean(acc_per_fold))
    print("average loss = ", np.mean(loss_per_fold))

#------------------Plot our accuracy and loss ----------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model results training history')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['Acuuracy', 'Loss'], loc='upper right')
plt.show()

#------------------Save and load model------------------------------
def save_model(model,model_name = "model"):
    model.save(model_name)
    
def load_model(path):
    model = models.load_model(path)
    return model
#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=test_images.shape[0])
print('train_acc:', train_acc)
print('test_acc:', test_acc)

