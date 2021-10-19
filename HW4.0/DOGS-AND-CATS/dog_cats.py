# -*- coding: utf-8 -*-

## Dogs are better than Cats, just saying.

import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


original_dataset_dir = "/Users/XuWan/dogs-vs-cats/train"
train_dir = "/Users/XuWan/dogs-vs-cats/train"
test_dir = "/Users/XuWan/dogs-vs-cats/test1"



