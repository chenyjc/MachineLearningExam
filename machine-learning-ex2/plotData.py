# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:50:51 2017

@author: cheyongj

Plot the positive and negative examples on a
              2D plot, using the option 'k+' for the positive
              examples and 'ko' for the negative examples.

"""
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
import pylab as pl


#
# https://stackoverflow.com/questions/23451028/matplotlib-pyplot-vs-matplotlib-pylab
# using pylab is better for iPython:
# https://stackoverflow.com/questions/16849483/which-is-the-recommended-way-to-plot-matplotlib-or-pylab
#

#def plotData(X,y):
#    pos=np.where(y==1)
#    neg=np.where(y==0)
#    matplotlib.rcParams['axes.unicode_minus'] = False
#    fig, ax = plt.subplots()
#    ax.plot(X[pos,0],X[pos,1], 'k+')
#    ax.plot(X[neg,0],X[neg,1], 'yo')
#    ax.set(xlabel='Exam1 score', ylabel ='Exam2 score',
#       title='Exam1 Test Data')
#    ax.legend(('Admited','Not Admited'),loc='bottom right')
#    
#    plt.show()

def plotData(X, y):
    # Find Indices of Positive and Negative Examples
    pos = np.where(y == 1) 
    neg = np.where(y == 0)

    # Plot Examples
    pl.plot(X[pos, 0], X[pos, 1], 'k+')
    pl.plot(X[neg, 0], X[neg, 1], 'yo')
    
    return