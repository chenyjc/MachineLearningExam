# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from ex2_func import *

# load data
data = np.loadtxt(fname = 'ex2data1.txt', delimiter = ',')

arr_data = np.array(data)

arr_X = arr_data[:, [0, 1]]
arr_y = arr_data[:, [2]]

# ==================== Part 1: Plotting ====================
# plot
plotData(arr_X, arr_y)

# Labels and Legend
pl.xlabel('Exam 1 score')
pl.ylabel('Exam 2 score')

# Specified in plot order
pl.legend(('Admitted', 'Not admitted'), loc = 'best')
pl.show()

# ============ Part 2: Compute Cost and Gradient ============
# Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = np.shape(arr_X)

# Add intercept term to x and X_test
arr_X = np.column_stack((np.ones([m, 1]), arr_X))

# Initialize fitting parameters
initial_theta = np.zeros([n + 1, 1])

# Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, arr_X, arr_y)
print(cost)
print(grad)

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
[cost, grad] = costFunction(test_theta, arr_X, arr_y)
print(cost)
print(grad)

# ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#result = opt.fmin_bfgs(f=costFunc, x0=initial_theta, fprime=gradientFunc, args=(arr_X, arr_y), full_output=True, retall=True)
#result = opt.fmin_bfgs(f=costFunction, x0=initial_theta, args=(arr_X, arr_y), full_output=True, retall=True)
result = opt.fmin_tnc(func=costFunction, x0=initial_theta, args=(arr_X, arr_y), disp=5)

#theta = np.array([[-25.161], [0.206], [0.201]])
theta = result[0]
print(theta)
# Plot Boundary
plotDecisionBoundary(theta, arr_X, arr_y)

pl.xlabel('Exam 1 score')
pl.ylabel('Exam 2 score')
pl.show()

# ============== Part 4: Predict and Accuracies ==============
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid(np.dot([1, 45, 85], theta))
print(prob)
# Compute accuracy on our training set
p1 = predict(theta, arr_X)

# added for python to convert to np.array[100, 1]
p = p1.reshape(np.size(p1), 1)
accuracy = np.mean(p==arr_y) * 100

print(accuracy)
##############################################
