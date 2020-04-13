import numpy as np

def getCost(x, y, theta, lambda1):
    m = np.size(x,0)
    h_0 = np.dot(x,theta)
    cost = (1/(2*m))*(np.sum(np.square((h_0-y)))) + ((lambda1)/(2*m))*np.sum(np.square(np.vstack([[0],theta[1:]])))
    return cost

def gradient(x,y ,theta, lambda1):
    m = np.size(x,0)
    h_0 = np.dot(x,theta)
    grad = (1/m)*(np.sum(((h_0-y)*x),0)).reshape(np.size(theta,0),1) + (lambda1/m)*np.sum(np.vstack([[0],theta[1:]]))
    return grad
