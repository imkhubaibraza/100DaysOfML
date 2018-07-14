#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:21:40 2017

@author: tauseef
"""
import numpy as np

def featureNormalization(x):


    mu1=np.mean(x[:, :1])
    mu2=np.mean(x[:, 1:])
    
    sigma1 =np.std(x[:, :1])
    sigma2=np.std(x[:, 1:])
    
   
    x1 = (x[:, :1] - mu1) / sigma1
    x2 = (x[:, 1:] - mu2) / sigma2
    
    x=np.append(x1,x2,axis=1)
    mu=np.append(mu1,mu2)
    sigma=np.append(sigma1,sigma2)

    
    return mu,sigma,x
