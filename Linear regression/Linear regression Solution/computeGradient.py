#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 14:12:36 2017

@author: tauseef
"""
import numpy as np
from costFunction import costFunction
def computeGradient(x,y,theta,alpha,iterations):
    J=[None];
    for i in range(iterations-1):
        # compute error here
        error = 0
        m=len(y)
        # write your code here to implement gradient descent dot(...) function
        # can be help full to find matrix multiplication in numpy. Implement it 
        # in such a general way that it can be used with multivariate Linear
        # Regression.
        hypothesis=np.dot(x,theta)
        error=hypothesis-y
        
        xTrans=np.transpose(x);
        x_mat=np.dot(xTrans,error)
        gradient=(alpha/m)*x_mat
        theta=theta-gradient
        
        
        
        
        
        
        
        # each time you compute theta following code call costFunction to get
        # cost with newtheta.
        J.append(costFunction(x,y,theta))
    return J,theta
    