'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from numpy.random import rand
from numpy.linalg import norm

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        
        n, d = X.shape
        cost = (-y.T*np.log(self.sigmoid(X*theta))-((1.0-y).T*np.log(1.0-self.sigmoid(X*theta))))+(regLambda/(2.0)*(theta.T*theta))
        return cost[0,0]
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        
        n, d = X.shape
        costGradient = ((X.T * (self.sigmoid(X*theta) - y) + regLambda * theta))
        costGradient[0] = (sum(self.sigmoid(X*theta) - y))
        return costGradient

    def sigmoid(self, Z):
        '''
        Computers the sigmoid function 1/(1+exp(-z))
        '''
        
        sm = 1.0/(1.0 + np.exp(-Z))
        return sm

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''

        n, d = X.shape
        X = np.c_[np.ones((n,1)), X]
        a = np.copy(d+1)
        mean = 0
        iterator = 1
        self.theta = np.mat(rand(a,1))
        oldTheta = self.theta
        newTheta = self.theta
        while iterator <= self.maxNumIters:
            newTheta = oldTheta - (self.alpha * self.computeGradient(newTheta,X,y,self.regLambda))
            hasConverged = norm(newTheta-oldTheta) < self.epsilon
            if (hasConverged != True):
                iterator += 1
                oldTheta = np.copy(newTheta)
            else:
                self.theta = newTheta
                return
        self.theta = newTheta

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        
        n, d = X.shape
        X = np.c_[np.ones((n,1)), X]
        predictions = np.array(self.sigmoid(X*self.theta))
        predictions[predictions>0.5] = 1
        predictions[predictions<0.5] = 0
        return predictions
                
#     def hasConverged(self, oldTheta, newTheta):
#         '''
#         Implementing a dedicated method as suggested in homework file
#         '''
#         return (norm(newTheta-oldTheta) < self.epsilon)