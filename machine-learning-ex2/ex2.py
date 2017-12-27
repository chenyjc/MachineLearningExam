# -*- coding: utf-8 -*-

import os
import numpy as np
from math import *
import scipy.optimize as opt

#from numpy import *

from costFunction import *
from plotData import *

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.
#load txt
data1=np.loadtxt('ex2/ex2data1.txt',delimiter=',')

X=data1[:,[0,1]]
y=data1[:,[2]]
print('-------X---------------')
print(X)
print('-------y---------------')
print(y)
plotData(X,y)
print('\nProgram paused. Press enter to continue.\n');
os.system("pause")

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = X.shape;

# Add intercept term to x and X_test
X = np.append(np.ones((m, 1)), X, axis=1);

# Initialize fitting parameters
initial_theta = np.zeros([n + 1, 1]);

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y);
grad = gradient(initial_theta, X, y);
#[cost, grad] = costFunction(initial_theta, X, y);

print('Cost at initial theta (zeros): %f\n'  %(cost));
print('Expected cost (approx): 0.693\n');
print('Gradient at initial theta (zeros): \n');
print(grad);
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');
print('\nProgram paused. Press enter to continue.\n');
os.system("pause")

## ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
#options = optimset('GradObj', 'on', 'MaxIter', 400);

#  Run fminunc to obtain the optimal theta
#  This function will return theta and the cost 
#[theta, cost] = ...
#	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
 
##### minimize function #####
#result = opt.minimize(fun = costFunction, x0 = initial_theta,
#                     args = (X, y),
#                            method = 'TNC',
#                            jac = gradient);
#optimal_theta = Result.x;        

#result = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y), disp=5)
#theta = result[0]

result = opt.minimize(fun = costFunction, x0 = initial_theta,
                     args = (X, y),
                            method = 'TNC',
                            jac = gradient);
theta = result.x                     


# Print theta to screen
print('Cost at theta found by fminunc: %f\n', cost);
print('Expected cost (approx): 0.203\n');
print('theta: \n');
print(' %f \n', theta);
print('Expected theta (approx):\n');
print(' -25.161\n 0.206\n 0.201\n');

# Plot Boundary
plotDecisionBoundary(theta, X, y);

print('\nProgram paused. Press enter to continue.\n');
os.system("pause")


## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.dot([1, 45, 85], theta));
print('For a student with scores 45 and 85, we predict an admission ', 
       'probability of \n', prob);
print('Expected value: 0.775 +/- 0.002\n\n');

# Compute accuracy on our training set
p = predict(theta, X);
           
p = p.reshape(np.size(p), 1)

print('p=:\n', p);
print('y=:\n', y);

print('Train Accuracy: %f\n' %(np.mean(p == y) * 100));
print('Expected accuracy (approx): 89.0\n');
print('\n');
