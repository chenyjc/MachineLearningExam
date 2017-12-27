# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:42:13 2017

@author: cheyongj
"""

import numpy as np
from sigmoid import *

def costFunctionReg(theta, X, y, v_lambda):
    
    #add for python code
    theta = theta.reshape((np.shape(X)[1],1))
  
    # Initialize some useful values
    m = np.size(y)    # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(np.shape(theta))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    
    
    g = sigmoid(np.dot(X, theta))
    J = (np.dot(np.transpose(y), np.log(g)) + np.dot(np.transpose(1-y), np.log(1 - g))) / (-m)
    grad = np.dot(np.transpose(X), (g-y)) / m
    
    # need to use copy() instead of theta_reg = theta in python
    theta_reg = theta.copy()
    theta_reg[0] = 0
    
  
    J_reg = (np.dot(np.transpose(theta_reg), theta_reg) * v_lambda) / (2*m)
    J = J + J_reg
    
    grad_reg = v_lambda * theta_reg / m
    grad = grad + grad_reg
    
    # =============================================================
  
    return(J, grad)