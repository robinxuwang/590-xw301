# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 14:09:36 2021

@author: xzw00
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json

#USER PARAMETERS
IPLOT=True
INPUT_FILE='LectureCodes/weight.json'
FILE_TYPE="json"
DATA_KEYS=['x','is_adult','y']
OPT_ALGO='BFGS'

#UNCOMMENT FOR VARIOUS MODEL CHOICES (ONE AT A TIME)
model_type = "logistic"; NFIT=4; xcol=1; ycol=2;

#READ FILE
with open(INPUT_FILE) as f:
    my_input = json.load(f)  #read into dictionary

#CONVERT INPUT INTO ONE LARGE MATRIX (SIMILAR TO PANDAS DF)
X=[];
for key in my_input.keys():
    if(key in DATA_KEYS): X.append(my_input[key])

#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
X=np.transpose(np.array(X))

#SELECT COLUMNS FOR TRAINING 
x=X[:,xcol];  y=X[:,ycol]

#COMPUTE BEFORE PARTITION AND SAVE FOR LATER
XMEAN=np.mean(x); XSTD=np.std(x)
YMEAN=np.mean(y); YSTD=np.std(y)

#NORMALIZE
x=(x-XMEAN)/XSTD;  y=(y-YMEAN)/YSTD; 

#PARTITION
f_train=0.8; f_val=0.2
rand_indices = np.random.permutation(x.shape[0])
CUT1=int(f_train*x.shape[0]); 
train_idx,  val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
xt=x[train_idx]; yt=y[train_idx]; xv=x[val_idx];   yv=y[val_idx]

#MODEL
def model(x,p):
    return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.01))))

#SAVE HISTORY FOR PLOTTING AT THE END
iter=0; iterations=[]; loss_train=[];  loss_val=[]

def predict(p):
    global YPRED_T,YPRED_V,MSE_T, MSE_V
    YPRED_T=model(x[train_idx],p)
    YPRED_V=model(x[val_idx],p)
    MSE_T=np.mean((YPRED_T-y[train_idx])**2.0) 
    MSE_V=np.mean((YPRED_V-y[val_idx])**2.0)

#LOSS FUNCTION
def loss(p, tidx_use, vidx_use):
    global iterations,loss_train,loss_val,iter

    #LOSS Function - train
    yp=model(x[tidx_use],p) #model predictions for given parameterization p
    training_loss=(np.mean((yp-y[tidx_use])**2.0))  #MSE
    
    #LOSS Function -validation
    yp=model(x[vidx_use],p) #model predictions for given parameterization p
    validation_loss=(np.mean((yp-y[vidx_use])**2.0))
    
    #RECORD FOR PLOTING
    loss_train.append(training_loss)
    loss_val.append(validation_loss)
    iterations.append(iter); iter+=1

    return training_loss

def a_loss(p, idx_use,result_list):
    global iterations,loss_train,loss_val,iter

    #LOSS Function 
    yp=model(x[idx_use],p) #model predictions for given parameterization p
    loss=(np.mean((yp-y[idx_use])**2.0))  #MSE

    return loss

#INITIAL GUESS
po=np.random.uniform(0.1,1.,size=NFIT)
pv=np.random.uniform(0.1,1.,size=NFIT)

#SAVE HISTORY FOR PLOTTING AT THE END
epoch=1; epochs=[]; loss_train=[];  loss_val=[]

def optimizer(f, po):
    global epoch
    
    #set up initial parameters
    algo = 'GD'
    iter = 1
    dx = 0.0001
    max_iter = 50000
    LR = 0.001
    #method = 'batch'
    method = 'mini-batch'
    #method = 'stochastic'
    NDIM = len(po)
    mom = 0.03
    first_half_t =[]
    second_half_t =[]
    for i in range(len(train_idx)):
        if i %2 == 0:
            first_half_t.append(train_idx[i])
        else:
            second_half_t.append(train_idx[i])
    first_half_v =[]
    second_half_v =[]
    for i in range(len(val_idx)):
        if i %2 == 0:
            first_half_v.append(val_idx[i])
        else:
            second_half_v.append(val_idx[i])
    #optimization loop
    while (iter <= max_iter):
        if(method == 'batch'):
            if(iter == 1): 
                tidx_use = train_idx
                vidx_use = val_idx
            if(iter >1): epoch += 1
            
            df_dx = np.zeros(NDIM)
            for i in range(0,NDIM):
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = po - dX
                xp1=po + dX
                
                grad_i = (f(xp1,tidx_use,vidx_use)-f(xm1,tidx_use,vidx_use))/dx/2
                df_dx[i] = grad_i
            
            #take step based on optization method
            if (algo == 'GD'):
                xip1=po-LR*df_dx
            if (algo == 'MOM'):
                xip1=po-LR*df_dx + mom*dx  
            if(iter%1==0):
                predict(po)
                print(iter,"    ",epoch,"    ",MSE_T,"    ",MSE_V)                        
            po = xip1
            iter += 1
                                   
            
        if(method == 'mini-batch'):
    
            if(iter == 1):
                first_tidx_use = first_half_t
                first_vidx_use = first_half_v
                second_tidx_use = second_half_t
                second_vidx_use = second_half_v                     
            if(iter >1): epoch += 1
            
            df_dx = np.zeros(NDIM)
            for i in range(0,NDIM):
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = po - dX
                xp1=po + dX
                
                grad_i = (f(xp1,first_tidx_use,first_vidx_use)-f(xm1,first_tidx_use,first_vidx_use))/dx/2
                df_dx[i] = grad_i
            
            #take step based on optization method
            if (algo == 'GD'):
                xip1=po-LR*df_dx
            if (algo == 'MOM'):
                xip1=po-LR*df_dx + mom*dx  
            if(iter%1==0):
                predict(po)                               
            po = xip1        
            for i in range(0,NDIM):
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = po - dX
                xp1=po + dX
                
                grad_i = (f(xp1,second_tidx_use,second_vidx_use)-f(xm1,second_tidx_use,second_vidx_use))/dx/2
                df_dx[i] = grad_i
            
            #take step based on optization method
            if (algo == 'GD'):
                xip1=po-LR*df_dx
            if (algo == 'MOM'):
                xip1=po-LR*df_dx + mom*dx  
            if(iter%1==0):
                predict(po)
                
            po = xip1
            print(iter,"    ",epoch,"    ",MSE_T,"    ",MSE_V)
            iter += 1
            
            
        if(method == 'stochastic'):
            f = a_loss
            for i in range(0,NDIM):
                dX = np.zeros(NDIM)
                dX[i] = dx
                xm1 = pv - dX
                xp1= pv + dX
            for j in range(len(val_idx)):
                vidx_use = val_idx[j]
                f(xp1,vidx_use,loss_val)
                f(xm1,vidx_use,loss_val)
                
            df_dx = np.zeros(NDIM)
            #take step based on optization method
            if (algo == 'GD'):
                xip1=pv-LR*df_dx
            if (algo == 'MOM'):
                xip1=pv-LR*df_dx + mom*dx  
            if(iteration%1==0):
                predict(pv)
            
            for j in range(len(train_idx)):
                tidx_use = train_idx[j]
                
                df_dx = np.zeros(NDIM)
                for i in range(0,NDIM):
                    dX = np.zeros(NDIM)
                    dX[i] = dx
                    xm1 = po - dX
                    xp1=po + dX
                
                    grad_i = (f(xp1,tidx_use,loss_train)-f(xm1,tidx_use,loss_train))/dx/2
                    df_dx[i] = grad_i
            
            #take step based on optization method
                if (algo == 'GD'):
                    xip1=po-LR*df_dx
                if (algo == 'MOM'):
                    xip1=po-LR*df_dx + mom*dx  
                if(iter%1==0):
                    predict(po)
                    print(iter,"    ",epoch,"    ",MSE_T,"    ",MSE_V) 
            print('step3')                       
            po = xip1
            print(iter)
            iter += 1
            
    return po
        
popt = optimizer(loss, po)  
                        
print("OPTIMAL PARAM:",popt)

#PREDICTIONS
xm=np.array(sorted(xt))
yp=np.array(model(xm,popt))

#UN-NORMALIZE
def unnorm_x(x): 
    return XSTD*x+XMEAN  
def unnorm_y(y): 
    return YSTD*y+YMEAN 

#FUNCTION PLOTS
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(unnorm_x(xt), unnorm_y(yt), 'o', label='Training set')
    ax.plot(unnorm_x(xv), unnorm_y(yv), 'x', label='Validation set')
    ax.plot(unnorm_x(xm),unnorm_y(yp), '-', label='Model')
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.legend()
    plt.show()

#PARITY PLOTS
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(model(xt,popt), yt, 'o', label='Training set')
    ax.plot(model(xv,popt), yv, 'o', label='Validation set')
    plt.xlabel('y predicted', fontsize=18)
    plt.ylabel('y data', fontsize=18)
    plt.legend()
    plt.show()

#MONITOR TRAINING AND VALIDATION LOSS  
if(IPLOT):
    fig, ax = plt.subplots()
    ax.plot(iterations, loss_train, 'o', label='Training loss')
    ax.plot(iterations, loss_val, 'o', label='Validation loss')
    plt.xlabel('optimizer iterations', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.legend()
    plt.show()
    