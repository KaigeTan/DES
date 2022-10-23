# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:17:11 2022

@author: Jonny
"""
import numpy as np

def Next(State,action,AGV_1,AGV_2,AGV_3,AGV_4,AGV_5,SUP_IPSR,SUP_ZWSR):
    
    X1 = State[0]   #from 0 
    X2 = State[1]
    X3 = State[2]
    X4 = State[3] 
    X5 = State[4]
    X6 = State[5]
    X7 = State[6]
    
    
    X1_ = np.where(AGV_1[X1, :, action] == 1)
    if len(X1_[0]) == 0:
        X1_ = X1
    else:
        X1_ = X1_[0][0]
       
    X2_ = np.where(AGV_2[X2, :, action] == 1)
    if len(X2_[0]) == 0:
        X2_ = X2
    else:
        X2_ = X2_[0][0]
       
       
    X3_ = np.where(AGV_3[X3, :, action] == 1)
    
    if len(X3_[0]) == 0:
       X3_ = X3
    else:
       X3_ = X3_[0][0]
        
        
    X4_ = np.where(AGV_4[X4, :, action] == 1)
    if len(X4_[0]) == 0:
        X4_ = X4
    else:
        X4_ = X4_[0][0]
       
       
    X5_ = np.where(AGV_5[X5, :, action] == 1)
    if len(X5_[0]) == 0:
        X5_ = X5
    else:
        X5_ = X5_[0][0]
        
          
    X6_ = np.where(SUP_IPSR[X6, :, action] == 1)   
    X6_ = X6_[0][0]
    
    X7_ = np.where(SUP_ZWSR[X7, :, action] == 1)
    X7_ = X7_[0][0]
    
    State_ = [X1_, X2_, X3_, X4_, X5_, X6_, X7_]
    
    return(State_)
    
    
    
    
    
    
    
    
    
    