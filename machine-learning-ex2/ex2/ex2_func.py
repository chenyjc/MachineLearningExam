# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 00:28:20 2017

@author: qianx
"""

import numpy as np
import pylab as pl
import scipy.optimize as opt

###########################################################
def plotData(X, y):
    # Find Indices of Positive and Negative Examples
    pos = np.where(y == 1) 
    neg = np.where(y == 0)

    # Plot Examples
    pl.plot(X[pos, 0], X[pos, 1], 'k+')
    pl.plot(X[neg, 0], X[neg, 1], 'yo')
    
    return

###########################################################
def sigmoid(z):
    # You need to return the following variables correctly 
    g = np.zeros(np.shape(z))

    g = 1 / (1 + np.exp(-z))
    return (g)

###########################################################
def costFunction(theta, X, y):
     # added for Python minimize function
    print('costFunction: theta=', theta)
    theta = theta.reshape(X.shape[1],1) 
    print('costFunction, after reshape: theta=', theta)
    
    # Initialize some useful values
    # number of training examples
    m = np.size(y) 
    
    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(np.shape(theta))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    g = sigmoid(np.dot(X, theta))
    
    J = (np.dot(np.transpose(y), np.log(g)) + np.dot(np.transpose(1-y), np.log(1 - g))) / (-m)
    grad = np.dot(np.transpose(X), (g-y)) / m
    
    #print('cost = '); print(J)
    #print('gradient = '); print(grad)
    return(J, grad)

###########################################################
def mapFeature(X1, X2):
    degree = 6 + 1
    out = np.ones(np.size(X1))
    
    for i in tuple(range(1, degree)):
        for j in tuple(range(0, i+1)):
            t = np.array(X1 ** (i-j) * (X2 ** j))
            out = np.column_stack((out, t))
    return(out)

###########################################################
def plotDecisionBoundary(theta, X, y):
    # Plot Data
    plotData(X[:, 1:3], y)
    
    if np.shape(X)[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = [np.min(X[:, 1])-2, np.max(X[:, 1])+2]
        # Calculate the decision boundary line
        plot_y = np.multiply(np.divide(np.add(np.multiply(plot_x, theta[1]), theta[0]), theta[2]), -1)
    
        pl.plot(plot_x, plot_y)
        
        pl.legend(('Admitted', 'Not admitted', 'Decision Boundary'))
        pl.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        len_u = np.size(u)
        len_v = np.size(v)
        
        z = np.zeros((len_u, len_v))
        
        # Evaluate z = theta*x over the grid
        for i in range(len_u):
            for j in range(len_v):
                z[i,j] = np.dot(mapFeature(u[i], v[j]), theta)
        
        z = np.transpose(z)    # important to transpose z before calling contour
        
        #X,Y = np.meshgrid(u, v)
        #pl.contour(X, Y, z, 8, colors = 'black', linewidth = 0.5)
        pl.contour(u, v, z, 0, colors = 'black', linewidth = 0.5)
    
    return

###########################################################
def predict(theta, X):
    # Number of training examples
    size = np.shape(X[:, 0]) 

    # You need to return the following variables correctly
    p = np.zeros([size[0], 1])

    p = np.round(sigmoid(np.dot(X, theta)))
    
    return(p)
    
###########################################################

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

###########################################################