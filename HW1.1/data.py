# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:36:40 2021

@author: xzw00
"""

# Constant Uppercase: CONSTANT, MY_CONSTANT, MY_LONG_CONSTANT
# Function: Lowercase: e.g function, my_function
# Variable: Lowercase: x, var, my_variabl

#--------------------------------
#import packages
#--------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

class Data:


#--------------------------------
#Load Data (Defining the problem and assembling a dataset)

#xlabel -- age
#ylabel -- weight
#--------------------------------    
    def _init_(self, file):
        self.file = 'weight.json'
        self.data = pd.read_json(self.file)
    
#--------------------------------
#Choosing a measure of success 
#--------------------------------

#In this case MSE is used as the loss function
    def loss(p):
        weight = p[0]
        bias = p[1]
        global X, Y    
        loss = []    
        preds = weight * X + bias   
        for pred, y in zip(preds, Y):
            loss.append((pred - y) ** 2)
        mse = sum(loss) / len(X)
        return mse   
#--------------------------------
#Test - Train Split (Deciding on an evaluation protocol)
#--------------------------------
    def split_lr(df): 
        x_train, x_test, y_train, y_test = train_test_split(df['x'], df['y'], test_size = 0.2)
        return x_train,x_test,y_train,y_test
    
    def split_logit(df): 
        #response variable is the binary variable is_adult
        #weight becomes feature
        x_train, x_test, y_train, y_test = train_test_split(df['y'], df['is_adult'], test_size = 0.2)
        return x_train,x_test,y_train,y_test



        #x = df['x']
        #y = df['y']
        #y_bi = df['is_adult']
        #x_train, x_test, y_train, y_test = train_test_split(df[p1], df[p2], test_size = 0.2)


#--------------------------------
#Normalize Data function (Preparing the data)
#--------------------------------
    def norm(x, y):    
        x_mean = x.mean()
        y_mean = y.mean()
        x_std = x.std()
        y_std = y.std()    
        x_norm = (x - x_mean)/x_std
        y_norm = (y - y_mean)/y_std    
        return x_norm, y_norm

    def de_norm(x, y):    
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)   
        x_de_norm= x*x_std +x_mean
        y_de_norm= y*y_std +y_mean   
        return x_de_norm, y_de_norm
#--------------------------------
#Developing models 
#--------------------------------
#    def M(x,p):
#        p=[b,m]=[p[0],p[1]]
 #       model = p[0]+p[1]*x
 #       return model 
#FUNCTION TO OPTIMZE
#    def lr_m(x,p):
#        out= p[0]*x + p[1]
#        return out

#In this case MSE is used as the loss function
    def optimize():
    #TRAIN MODEL USING SCIPY OPTIMIZER
        res = minimize(loss, [0,0], method='Nelder-Mead', tol=1e-15)
        popt=res.x
        #print("OPTIMAL PARAM:",popt)
    #
        x_de_norm, yp_de_norm = de_norm(x_train_norm, x_train_norm*popt[0] +popt[1])
        plt.scatter(x_train, y_train)
        plt.plot(x_de_norm, yp_de_norm, color='r')
        plt.show()
        return popt

    def ln_regression(x,y):
        #only fit to age < 18 per the prompt        
        df_18 = df[df['x'] < 18]
        #train-test split
        x_train,x_test,y_train,y_test = split_lr(df_18)
        
        #normalization
        x_norm, y_norm = norm(x_train, y_train)
        
        
        
        
        
        
        
        
        
    
    def logit_regression():
        
    def logit_classification():
        



#FUNCTION TO OPTIMZE
    def f(x):
        =x**2.0
        ut=(x+10*np.sin(x))**2.0
        urn out

def f1(x): 
	global num_func_eval
	out=f(x)
	num_func_eval+=1
	if(num_func_eval%10==0):
		print(num_func_eval,x,out)
	plt.plot(x,f(x),'ro')
	plt.pause(0.11)
	return out

#INITIAL GUESS 
xo= 10 #
#xo=np.random.uniform(xmin,xmax)
print("INITIAL GUESS: xo=",xo, " f(xo)=",f(xo))
res = minimize(f1, xo, method='Nelder-Mead', tol=1e-5)
popt=res.x
print("OPTIMAL PARAM:",popt)

plt.show()