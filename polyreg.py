'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
from numpy.linalg import pinv
from numpy import dot
from numpy import c_
from numpy import ones

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        self.mean = None
        self.std = None
        self.theta = None
        self.degree = degree
        self.regLambda = regLambda


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        #self.degree = degree
        #self.X = X
        #polyRegArr = []
        #polyRegArr = np.copy(X)
        
        #for i in xrange(self.X.size):
        #i = 2
        #j = degree+1
        #for k in xrange(i, j):
        
            #temp = []
            
            #for j in xrange(self.degree):
            #for j in xrange(degree):
            
                #k = j + 1
                #temp.append(self.X[i] ** k)
                #temp.append(X[i] ** k)
            #polyRegArr.append(temp)
        polyRegArr = np.copy(X)
        i = 2
        j = degree+1
        for k in range(i, j):
            polyRegArr = c_[polyRegArr, X**k]
        return polyRegArr

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        #tempDeg = self.degree
        ndArray = self.polyfeatures(X, self.degree)
        n = ndArray.shape[0]
        
        #ndMatrixNew = array(ndArray)
        mean = np.mean(ndArray, axis = 0)
        std = np.std(ndArray, axis = 0)
        self.mean = mean
        self.std = std
        #temp1 = ndArray - mean
        ndMatrixNew = (ndArray-self.mean)/std
        s = (n,1)
        ndMatrixNew = c_[ones(s), ndMatrixNew]
        #n = None
        #d = None
        #n, d = ndMatrixNew.shape
        ndMatrixFinal = np.identity(self.degree+1) #* self.regLambda
        ndMatrixFinal[0,0] = 0
        self.theta = dot(dot(pinv(dot(ndMatrixNew.T, ndMatrixNew) + self.regLambda*ndMatrixFinal), ndMatrixNew.T), y)
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        n1Array = self.polyfeatures(X, self.degree)
        n = n1Array.shape[0]
        n1ArrayNew = (n1Array-self.mean)/self.std
        s = (n,1)
        n1ArrayNew = c_[ones(s), n1ArrayNew]
        return dot(n1ArrayNew, self.theta)


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------



def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    
    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    model = PolynomialRegression(degree=degree,regLambda=regLambda)
    for i in xrange(2,n+1):
        model.fit(Xtrain[0:i], Ytrain[0:i])
        trainingPred = model.predict(Xtrain[0:i])
        errorTrain[i-1] = sum((trainingPred - Ytrain[0:i])**2) * 1.0/i
        testPred = model.predict(Xtest)
        errorTest[i-1] = sum((testPred - Ytest)**2) * 1.0/len(Xtest)
    return (errorTrain, errorTest)
