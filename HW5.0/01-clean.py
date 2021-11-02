# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 22:44:08 2021

@author: XuWan
"""
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
# load text
novels = ['the_game_of_go.txt','the_golden_chimney.txt','honor_of_thieves.txt']

#novels = ['the_game_of_go.txt']

# imdb_dir = '/Users/fchollet/Downloads/aclImdb'
# train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []

for label_type in ['the_game_of_go', 'the_golden_chimney','honor_of_thieves']:
    fname = label_type + '.txt'
    with open(fname,'r', encoding="utf8") as f:
        raw = f.read()     
        f.close()
        raw = sent_tokenize(raw)
    i = 0
    for sentence in raw:
        if i > 4 and i < 2505:    #skip first few sentences since they're almost the same for all books
            #basic text claining 
            sentence = sentence.lower()
            sentence = re.sub(r"[^A-Za-z\-\n']", " ", sentence)
            sentence = re.sub("  ", " ", sentence)
            sentence = re.sub(' +', ' ', sentence)
            sentence = re.split('\r|\n', sentence)
            sentence = [x for x in sentence if x]
            sentence = ''.join(sentence)
           # print(sentence)
            texts.append(sentence)              
            if label_type == 'the_game_of_go':
                labels.append(0)
            elif label_type == 'the_golden_chimney':
                labels.append(1)
            else:
                labels.append(2)
        i += 1

##write the data to a txt file for next step
with open("texts.txt", "w") as f_x:
    for line in texts:
        f_x.write(line +"\n")

## save labels 
with open("labels.txt", "w") as f_y:
    for line in labels:
        f_y.write(str(line) +"\n")  


