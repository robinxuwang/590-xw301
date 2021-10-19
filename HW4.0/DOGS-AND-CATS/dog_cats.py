# -*- coding: utf-8 -*-

#-----------------------------------------
# Dogs are better than Cats, just saying.
#-----------------------------------------
import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image



#--------------------------------------------------
#Load Data and create a smaller dataset
#--------------------------------------------------


#create folders to store a smaller version of the data
original_dataset_dir = "/Users/XuWan/dogs-vs-cats/train"
base_dir = "/Users/XuWan/cats_and_dogs_small"
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)


#copy files into the folder
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = original_dataset_dir + '/'+ fname
    dst = train_cats_dir + '/' + fname
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = original_dataset_dir + '/'+ fname
    dst = validation_cats_dir + '/'+ fname
    shutil.copyfile(src, dst)
   
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = original_dataset_dir + '/'+ fname
    dst = test_cats_dir + '/'+ fname
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = original_dataset_dir + '/'+ fname
    dst = train_dogs_dir + '/'+ fname
    shutil.copyfile(src, dst)
    
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = original_dataset_dir + '/'+ fname
    dst = validation_dogs_dir + '/'+ fname
    shutil.copyfile(src, dst)
    
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = original_dataset_dir + '/'+ fname
    dst = test_dogs_dir + '/'+ fname
    shutil.copyfile(src, dst)   
    
    
#-------------------------------------------------
#   Model building 
#-------------------------------------------------    

#initialize sequential model
model = models.Sequential()

#add layers to the model
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150)
                                                    ,batch_size=20,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150)
                                                    ,batch_size=20,class_mode='binary')



history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=validation_generator,
                              validation_steps=50)

#save the model
model.save('cats_and_dogs_small_1.h5')

#---------------------------------------------------------
# plot the parameters
#---------------------------------------------------------
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



#-------------------------------------------------
#skips data augmaentation and feature extraction
#-------------------------------------------------

# Visualization 
img_path =  original_dataset_dir + '/cat.1.jpg'
img = image.load_img(img_path,target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis = 0)
img_tensor /= 255 


plt.imshow(img_tensor[0])
plt.show()


layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
layer_names = []
for layer in model.layers[:4]:
    layer_names.append(layer.name)

images_per_row = 8
for layer_name, layer_activation in zip(layer_names, activations):
    #number of features in the feature map
    n_features = layer_activation.shape[-1]
    # The feature map has shape (1,size,size,number of features)
    size = layer_activation.shape[1]
    # tiles the activation channels in this matrix 
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    #Tiles each filter into a big horizontal grid 
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row+row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size: (col + 1) * size,
                         row*size: (row + 1)*size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale*display_grid.shape[1], scale*display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()







