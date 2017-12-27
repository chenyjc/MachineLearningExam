# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:42:13 2017

@author: cheyongj
"""

import numpy as np
import pylab as pl
from plotData import *
from mapFeature import *

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