#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:56:03 2017

@author: tauseef
"""
import numpy as np
from numpy.linalg import inv
def numericalGradient(x,y):
		# Here you have to provide code to compute design matrix D = (X^T X)^{-1}	

    D = np.linalg.inv(np.dot(np.transpose(x),x))
	# Here you have to compute D * X^T y
   
    theta = np.dot((np.dot(y,x)),D)
    return theta
