# -*- coding: utf-8 -*-
"""
Author: Tauseef Iftikhar

Date: 20-10-2017
"""
""" 
numpy is a package to support n-dimensional homogeneous 
arrays. For further details on numpy visit the link 
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
"""
import numpy as np

"""
matplotlib is a package use to draw plots of high quality
python. For further details on matplotlib visit the link
https://matplotlib.org/
"""
import matplotlib.pyplot as mp
"""
panda is a package to support dealing data. It is used to
represent and manipulate data in tabular form. For further
details on matplotlib visit the link
https://pandas.pydata.org/pandas-docs/stable/tutorials.html
"""
import pandas as pd

from multi_cost_function import multiCostFunction
from multi_compute_gradient import computeMultiGradient
from feature_normalization import featureNormalization
from numerical_gradient import numericalGradient

# reading a comma separated tedt file.
# df = pd.read_csv('univariate_linear_regression_data.txt')
df = pd.read_csv('assign2data2.txt')
# following command converts a dataframe into an numpy array.
data = df.as_matrix() 
x = data[:,:-1]

# moving last column to y
y = data[:,-1]

m = len(y)

print('x is',x);

"""Creating new features
you have to provide code in following line
"""
xx=np.power(x,2)
xx2=np.column_stack((x,xx))
xx=np.power(x,4)
new_x = np.column_stack((xx2,xx))
"""
feature normalization

perform feature normalization if there are multiple feature in x

"""

mu,sigma,X_norm = featureNormalization(new_x)
# plotting data on 2D plane
mp.plot(x,y,'o')

# creating a numpy array(vector) of 1's of length N
ones = np.ones(m)
# adding column of 1's in front of x (Note: each 1 is representing
# x_o in an example of input training data.) 
X = np.column_stack((ones,X_norm))
#initialize theta with 0's (Size of theta should be equal to
#number of number of attributes in x in training examples.)
N = X.shape[1]

theta = np.zeros(N)
"""
You are going to implement the costFunction(...) called in 
next line.
"""
print('cost is= ', multiCostFunction(X,y,theta))
# if you have implemented costFunction correctly then cost 
#should be 65279190347.9
alpha = .005;
iterations = 100000;
"""
You are going to implement the computeGradient(...) called in 
next line.
"""
cost,theta = computeMultiGradient(X,y,theta,alpha,iterations)

print('Theta is= ', theta)

print('minimum cost is: ', multiCostFunction(X,y,theta))
mp.figure(0)
mp.plot(x,y,'o')
mp.plot(x,X.dot(theta),'-')
"""
Following lines draw the number of iterations against the cost
for each iterations.
"""
it = np.array(range(len(cost)))

fig, debug = mp.subplots()
debug.plot(it,cost,'-') 
debug.set(xlabel='number of iterations', ylabel='cost', title='Debugging')


