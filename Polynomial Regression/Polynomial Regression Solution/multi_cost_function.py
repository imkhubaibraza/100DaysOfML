#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 14:04:39 2017

@author: tauseef
"""
import numpy as np

def multiCostFunction(X,y,theta):
    m = len(X)
    hyp = np.dot(theta,np.transpose(X))
    error =hyp-y 
    cost = error.dot(np.transpose(error))
    cost = cost/(2 * m)
    return cost
