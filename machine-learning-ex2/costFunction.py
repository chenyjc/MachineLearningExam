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

import math

def costFunction(theta, X, y):
    m = y.shape[0]
    
    htheta = sigmoid(np.dot(X, theta))
    J = 1 / m * np.sum(-y * math.log(htheta) - (1-y) * math.log(1 - htheta))
    
    grad = []
    for i in range(0,theta.shape[0]):
        grad[i]= 1 / m * np.sum((htheta - y) * X[:,i])
    
    return J,grad