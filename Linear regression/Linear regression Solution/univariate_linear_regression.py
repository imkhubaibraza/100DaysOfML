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

from costFunction import costFunction
from computeGradient import computeGradient
from plotData import plotData
# reading a comma separated text file.
df = pd.read_csv('univariate_linear_regression_data.txt')
# moving first column  lableled 'x' in variable x as a Series
x = df['x']
# moving first column lableled 'y' in variable y as a Series
y = df['y']

M = len(x)
# plotting data on 2D plane. Provide the code in plotData file.
plotData(x,y)

# creating a numpy array(vector) of 1's of length N
ones = np.ones(M)
# adding column of 1's in front of x (Note: each 1 is representing
# x_o in an example of input training data.) 
new_x = np.column_stack((ones,x))
#initialize theta with 0's (Size of theta should be equal to
#number of attributes in x in training examples.)
N = new_x.shape[1]
theta = np.zeros(N)
"""
You are going to implement the costFunction(...) called in 
next line.
"""
print('cost is: ', costFunction(new_x,y,theta))

print('correct cost is: 2.550155155868016' )
# if you have implemented costFunction correctly then cost 
#should be 2.550155155868016
alpha = 0.01;
iterations = 5000
"""
You are going to implement the computeGradient(...) called in 
next line.
"""
cost,theta = computeGradient(new_x,y,theta,alpha,iterations);

"""
Following lines draw the number of iterations against the cost
for each iterations.
"""
it = np.array(range(len(cost))) ;

fig, debug = mp.subplots()
debug.plot(it,cost,'-') 
debug.set(xlabel='number of iterations', ylabel='cost', title='Debugging')

print('new theta is: ', theta);

print('correct theta is:  [1.98901186  0.52657856]')

# if you have implemented computeGradient correctly then new theta 
#should be [ 1.87806909  0.73357442]

"""
Following lines of code draw the line obtained using learned 
theta on the data set.
"""
mp.figure(0);
mp.plot(x,y,'o')
mp.plot(x,new_x.dot(theta),'-');