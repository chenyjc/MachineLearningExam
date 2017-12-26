# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:52:13 2017

@author: cheyongj

Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).

"""
import math

def sigmoid(z):
    return 1 / (1+math.exp(-z))