import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    flag = 6
    i = 1
    feature = 1
    n = x1.shape[0]
    nt = ((flag+1)*(flag+2))/2
    expFeature = np.ones((n, nt))
    for j in range(i, flag+1):
        for k in range(j+1):
            expFeature[:,feature] = (x1**(j-k)) * (x2**k)
            feature = feature+1
    return expFeature