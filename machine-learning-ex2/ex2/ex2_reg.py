# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:07:18 2017

@author: qianx
"""

from ex2_func import *


## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

# load data
data = np.loadtxt(fname = 'ex2data2.txt', delimiter = ',')

arr_data = np.array(data)

X = arr_data[:, [0, 1]]
y = arr_data[:, [2]]

plotData(X, y)
# Labels and Legend
pl.xlabel('Microchip Test 1')
pl.ylabel('Microchip Test 2')

# Specified in plot order
pl.legend(('y = 1', 'y = 0'), loc = 'best')
pl.show()

## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic
#  regression to classify the data points.
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:,0], X[:,1])

# Initialize fitting parameters
initial_theta = np.zeros((np.size(X, 1), 1))

# Set regularization parameter lambda to 1
v_lambda = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
[cost, grad] = costFunctionReg(initial_theta, X, y, v_lambda)
print('cost ='); print(cost)
print('grad [1th..5th] = '); print(grad[0:5,])

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((np.shape(X)[1],1))
[cost, grad] = costFunctionReg(test_theta, X, y, 10)
print('cost ='); print(cost)
print('grad [1th..5th] = '); print(grad[0:5,])


## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#

# Initialize fitting parameters
initial_theta = np.zeros((np.shape(X)[1],1))

# Set regularization parameter lambda to 1 (you should vary this)
v_lambda = 1

# Optimize
result = opt.fmin_tnc(func=costFunctionReg, x0=initial_theta, args=(X, y, v_lambda), disp=5)

theta = result[0]

# Plot Boundary
plotDecisionBoundary(theta, X, y)
# Labels and Legend
pl.xlabel('Microchip Test 1')
pl.ylabel('Microchip Test 2')

# Specified in plot order
pl.legend(('y = 1', 'y = 0'), loc = 'best')
pl.show()


