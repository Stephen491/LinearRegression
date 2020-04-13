import numpy as np
from linear_regression import gradient
from linear_regression import getCost

def gradient_descent(x,y,theta,alpha,lambda1, iterations):
    m =  np.size(x,0)
    for i in range(iterations):
        theta = theta - alpha*gradient(x,y,theta,lambda1)
        #print(getCost(x, y, theta, lambda1))
    return theta