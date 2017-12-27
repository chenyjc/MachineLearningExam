# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:52:13 2017

@author: cheyongj

Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).

"""
import numpy as np

def sigmoid(z):
#    print('inside sigmoid z=:\n',z)
    g=np.zeros(np.shape(z))
    g=1 / (1+np.exp(-z))
#    print('inside sigmoid g=:\n',g)
    return (g)