# -*- coding: utf-8 -*-

import numpy as np

#load txt
data1=np.loadtxt('ex2/ex2data1.txt',delimiter=',')

X=data1[:,0:2]
y=data1[:,2]
plotData(X,y)

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = X.shape;

# Add intercept term to x and X_test
X = np.append(np.ones((m, 1)), X);

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1));

# Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

print('Cost at initial theta (zeros): %f\n'  %(cost));
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');