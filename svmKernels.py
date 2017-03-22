"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np
from numpy import dot
from numpy import zeros
from numpy import sum
from numpy import exp
from numpy import sqrt
from numpy import power

_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return ((dot(X1, X2.T)+1)** _polyDegree)



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    a = X1.shape[0]
    b = X2.shape[0]
    n = (a,b)
    squaredL2 = zeros(n)
    for i in xrange(b):
        squaredL2[:,i] = sum((X1-X2[i,:])**2,axis=1)
    return (exp(-squaredL2/(2*_gaussSigma**2)))



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    X1 = np.matrix(X1, copy=False)
    X2 = np.matrix(X2, copy=False)
    return np.asarray((dot(X1,X2.T)/(sqrt(sum(power(X1,2),axis=1)))/(sqrt(sum(power(X2,2),axis=1))).T))

