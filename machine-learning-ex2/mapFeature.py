# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:42:13 2017

@author: cheyongj
"""

import numpy as np

def mapFeature(X1, X2):
    degree = 6 + 1
    out = np.ones(np.size(X1))
    
    for i in tuple(range(1, degree)):
        for j in tuple(range(0, i+1)):
            t = np.array(X1 ** (i-j) * (X2 ** j))
            out = np.column_stack((out, t))
    return(out)