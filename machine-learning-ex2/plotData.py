# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:50:51 2017

@author: cheyongj

Plot the positive and negative examples on a
              2D plot, using the option 'k+' for the positive
              examples and 'ko' for the negative examples.

"""
import matplotlib
import matplotlib.pyplot as plt

def plotData(X,y):
    posX=X[y==1]
    negX=X[y==0]
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots()
    ax.plot(posX[:,0],posX[:,1], 'k+',label='Admited')
    ax.plot(negX[:,0],negX[:,1], 'yo',label='Not Admited')
    ax.set(xlabel='Exam1 score', ylabel ='Exam2 score',
       title='Exam1 Test Data')
    ax.legend(loc='upper right')
    
    plt.show()
    pass