#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 14:12:36 2017

@author: tauseef
"""
import numpy as np
from multi_cost_function import multiCostFunction
def computeMultiGradient(x,y,theta,alpha,iterations):
    m = len(y)
    J=[]
    
    for i in range(iterations-1):
	
        error =  np.dot(x,np.transpose(theta))-y
        theta = theta-(alpha/m)*(np.dot(np.transpose(x),error))
        J.append(multiCostFunction(x,y,theta))
    return J,theta
