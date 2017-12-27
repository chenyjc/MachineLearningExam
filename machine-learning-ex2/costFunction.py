# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:56:38 2017

@author: cheyongj

COSTFUNCTION Compute cost and gradient for logistic regression
   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
   parameter for logistic regression and the gradient of the cost
   w.r.t. to the parameters.
   
Note: grad should have the same dimensions as theta   

Return [J, grad]

"""

import numpy as np

from sigmoid import *

def costFunction(theta, X, y):
    #print('costFunction: shape of theta:', theta.shape)
    #print('costFunction: theta:', theta)
    theta = theta.reshape(X.shape[1], 1)
    #print('costFunction: after reshape - theta:', theta)
    m = y.shape[0]
    
    z=np.dot(X, theta)
    htheta = sigmoid(z)

    J = (np.dot(np.transpose(y), np.log(htheta)) + np.dot(np.transpose(1-y) , np.log(1 - htheta))) / (-m)
    #grad= np.dot(np.transpose(X), (htheta - y)) / m
    
    return (J)
    #return (J,grad)

def gradient(theta, X, y):
    m = y.shape[0]
    
    theta = theta.reshape(X.shape[1], 1)
    
    z=np.dot(X, theta)
    htheta = sigmoid(z)

    grad= np.dot(np.transpose(X), (htheta - y)) / m
    grad = np.transpose(grad)                
    
    return (grad)