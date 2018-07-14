#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:00:29 2017

@author: tauseef
"""
import numpy as np

def predict(x, theta):
    ones_x=np.ones(1);
    #make aaray with 1x3 to dot product with thetha
    xx = np.concatenate((ones_x,x)) 
    
    return np.dot(theta,np.transpose(xx))
