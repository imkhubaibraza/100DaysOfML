import numpy as np
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 14:04:39 2017

@authoor: tauseef
"""

def costFunction(x,y,theta):
    m = len(y)
    cost = 0
    # Compute hypothesis bellow
    hypothesis=np.dot(x,theta)
    
    # Compute Error term here...
    error=hypothesis-y
    
    
    # Compute cost be dot product of Error with itself and then find mean.
    value_s=np.dot(np.transpose(error),error)
    cost=value_s/(2*m)
    
    return cost