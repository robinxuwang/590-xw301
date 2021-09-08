# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:04:48 2021

@author: xzw00
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


class Data:

    def norm(x, y):    
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_std = np.std(x)
        y_std = np.std(y)    
        x_norm = (x - x_mean)/x_std
        y_norm = (y - y_mean)/y_std    
        return x_norm, y_norm
    
    def split_lr(x,y):    
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
        return x_train,x_test,y_train,y_test
    

    def optimize_lr():
        def loss(p):
            weight = p[0]
            bias = p[1]
            global x_train, y_train   
            loss = []    
            preds = weight * x_train + bias   
            for pred, y in zip(preds, y_train):
                loss.append((pred - y) ** 2)
                mse = sum(loss) / len(x_train)
                return mse
        #TRAIN MODEL USING SCIPY OPTIMIZER
        res = minimize(loss, [0,0], method='Nelder-Mead', tol=1e-15)
        popt=res.x
        #print("OPTIMAL PARAM:",popt)        
        x_de_norm = x_train_norm*x_std_lr +x_mean_lr
        yp_de_norm = x_de_norm*popt[0] +popt[1]
        plt.scatter(x_train, y_train)
        plt.plot(x_de_norm, yp_de_norm, color='r')
        plt.xlabel('age')
        plt.ylabel('weight')
        plt.show()
        return popt
    
    def optimize_logit_classification():
        def loss(p):
            global x_train, y_train   
            loss = []    
            preds =p[0] +p[1]*(1/1+np.exp(-(x_train-p[2])/p[3]+0.00001))
            for pred, y in zip(preds, y_train):
                loss.append((pred - y) ** 2)
                mse = sum(loss) / len(x_train)
                return mse
        #TRAIN MODEL USING SCIPY OPTIMIZER
        res = minimize(loss, [0,0,0,0], method='Nelder-Mead', tol=1e-15)
        popt=res.x
        #print("OPTIMAL PARAM:",popt)        
        x_de_norm = x_train_norm*x_std_logit +x_mean_logit
        yp_de_norm =popt[0] +popt[1]*(1/1+np.exp(-(x_de_norm-popt[2])/popt[3]+0.00001))
        plt.scatter(x_train, y_train)
        plt.plot(x_de_norm, yp_de_norm, color='r')
        plt.xlabel('weight')
        plt.ylabel('is_adult')
        plt.show()
        return popt
    
    def optimize_logit_reg():
        def loss(p):
            global x_train, y_train   
            loss = []    
            preds =p[0] +p[1]*(1/1+np.exp(-(x_train-p[2])/p[3]+0.00001))
            for pred, y in zip(preds, y_train):
                loss.append((pred - y) ** 2)
                mse = sum(loss) / len(x_train)
                return mse
        #TRAIN MODEL USING SCIPY OPTIMIZER
        res = minimize(loss, [0,0,0,0], method='Nelder-Mead', tol=1e-15)
        popt=res.x
        #print("OPTIMAL PARAM:",popt)        
        x_de_norm = x_train_norm*x_std_logitre +x_mean_logitre
        yp_de_norm =popt[0] +popt[1]*(1/1+np.exp(-(x_de_norm-popt[2])/popt[3]+0.00001))
        plt.scatter(x_train, y_train)
        plt.scatter(x_de_norm, yp_de_norm, color='r')
        plt.xlabel('age')
        plt.ylabel('weight')
        plt.show()
        return popt
    

if __name__ == '__main__': 
    df = pd.read_json('weight.json')
    df_lr = df[df['x']<18]
    x_mean_lr = np.mean(df_lr['x'])
    y_mean_lr = np.mean(df_lr['y'])
    x_std_lr = np.std(df_lr['x'])
    y_std_lr = np.std(df_lr['y'])
    p = [0,0]     
    x_train_norm, y_train_norm = Data.norm(df_lr['x'],df_lr['y'])
    x_train, x_test, y_train, y_test = Data.split_lr(df_lr['x'],df_lr['y'])
    Data.optimize_lr()
    x_mean_logit = np.mean(df['y'])
    y_mean_logit = np.mean(df['is_adult'])
    x_std_logit = np.std(df['y'])
    y_std_logit = np.std(df['is_adult'])
    p = [0,0,0,0]
    x_train_norm, y_train_norm = Data.norm(df['y'],df['is_adult'])
    x_train, x_test, y_train, y_test = Data.split_lr(df['y'],df['is_adult'])
    Data.optimize_logit_classification()
    x_mean_logitre = np.mean(df['x'])
    y_mean_logitre = np.mean(df['y'])
    x_std_logitre = np.std(df['x'])
    y_std_logitre = np.std(df['y'])
    p = [0,0,0,0]
    x_train_norm, y_train_norm = Data.norm(df['x'],df['y'])
    x_train, x_test, y_train, y_test = Data.split_lr(df['x'],df['y'])
    Data.optimize_logit_reg()
 
