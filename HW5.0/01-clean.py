# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 18:40:46 2021

@author: XuWan
"""

import numpy as np
import re
import os
from tensorflow.keras import utils

#novels = ['the_game_of_go']
novels = ['the_game_of_go','the_golden_chimney','honor_of_thieves']

    #split 
def split_n (listtemp, n):
    for i in range(0, len(listtemp), n):
        yield listtemp[i:i+n]

# for book in novels:
#     filename = book + '.txt'
#     file = open(filename, encoding="utf8")
#     text = file.read()
#     #print(text[500:700])
#     #basic text cleaning
#     text = re.sub(r"[^A-Za-z\-\n']", " ", text)
#     text = re.sub('\n', ' ', text)
#     text = re.sub('\d', '', text)
#     text = re.sub('  ', ' ', text) 
#     text = text.lower()
    
#     #split the txt file into smaller chunks
#     words_split = split_n(text, 250)
#     i = 0
#     for chunk in words_split:
#         if i < 600:
#             if i % 5 == 0:
#                 with open('./test/'+ book +'/'+ str(i) + '.txt', "w") as f:
#                     f.write(chunk)
#             else:
#                 with open('./train/'+ book +'/'+ str(i) + '.txt', "w")  as f:
#                     f.write(chunk)
#             i += 1
#     #print(text[500:700])
#     file.close()


#now we have three folders, each containing a train folder and a test folder 
#that are smaller chunks of each novel
train_labels = []
train_texts = [] 
train_dir = './train'


test_labels = []
test_texts = [] 
test_dir = './test'
   
for label_type in ['the_game_of_go','the_golden_chimney','honor_of_thieves']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            train_texts.append(f.read())
            f.close()
        if label_type == 'the_game_of_go':
            train_labels.append(0)
        elif label_type == 'the_golden_chimney':
            train_labels.append(1)
        else:
            train_labels.append(2)
for label_type in ['the_game_of_go','the_golden_chimney','honor_of_thieves']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            test_texts.append(f.read())
            f.close()
        if label_type == 'the_game_of_go':
            test_labels.append(0)
        elif label_type == 'the_golden_chimney':
            test_labels.append(1)
        else:
            test_labels.append(2)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train_labels)
ytrain = encoder.transform(train_labels)

encoder.fit(test_labels)
ytest = encoder.transform(test_labels)

## save texts for next step
with open("test_texts.txt", "w") as f_x1:
    for line in test_texts:
        f_x1.write(str(line) +"\n\n\t\n")
with open("train_texts.txt", "w") as f_x2:
    for line in train_texts:
        f_x2.write(str(line) +"\n\n\t\n")

## save labels 
with open("train_y.txt", "w") as f_y1:
    for line in ytrain:
        f_y1.write(str(line) +"\n\n\t\n")   
with open("test_y.txt", "w") as f_y2:
    for line in ytest:
        f_y2.write(str(line) +"\n\n\t\n")   


    
    
