import scipy
from pylab import *

import scipy
from pylab import *
import matplotlib.pyplot as plt


plt.figure(1)

pos = find(y==1); neg = find(y == 0);

fig, axs = plt.subplots(1,1,figsize=(18, 5))
axs.scatter(X[pos,0], X[pos,1])

plt.show()

def plotData(X, y):
    pos = find(y == 1);
    neg = find(y == 0);

    print(pos)
    print(neg)
    # plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
    # plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
    plot(X(pos, 1), X(pos, 2));
    plot(X(neg, 1), X(neg, 2));
    pass

def preditct():
    pass

def costFunction():
    pass

def sigmoid():
    pass

def mapFeature():
    pass

data=np.loadtxt("ex2data1.txt",delimiter=",")
X=data[:,0:2]
y=data[:,2]

plt.figure(1)

plotData(X, y)

#plt.plot(X,y)

plt.show()
