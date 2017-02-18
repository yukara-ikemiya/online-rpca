# -*- coding: utf-8 -*-
import numpy as np

def thres(x, mu):
    """
    y = sgn(x)max(|x| - mu, 0)
    
    Parameters
    ----------
    x: numpy array
    mu: thresholding parameter
    
    Returns:
    ----------
    y: numpy array
    
    """
    y = np.maximum(x - mu, 0)
    y = y + np.minimum(x + mu, 0)
    return y
    
def solve_proj2(m, U, lambda1, lambda2):
    """
    solve the problem:
    min_{v, s} 0.5*|m-Uv-s|_2^2 + 0.5*lambda1*|v|^2 + lambda2*|s|_1
    
    solve the projection by APG

    Parameters
    ----------
    m: nx1 numpy array, vector
    U: nxp numpy array, matrix
    lambda1, lambda2: tuning parameters
    
    Returns:
    ----------
    v: px1 numpy array, vector
    s: nx1 numpy array, vector

    """
    # intialization
    n, p = U.shape
    v = np.zeros(p)
    s = np.zeros(n)
    I = np.identity(p)
    converged = False
    maxIter = np.inf
    k = 0
    # alternatively update
    UUt = np.linalg.inv(U.transpose().dot(U) + lambda1*I).dot(U.transpose())
    while not converged:
        k += 1
        vtemp = v
        # v = (U'*U + lambda1*I)\(U'*(m-s))
        v = UUt.dot(m - s)       
        stemp = s
        s = thres(m - U.dot(v), lambda2)
        stopc = max(np.linalg.norm(v - vtemp), np.linalg.norm(s - stemp))/n
        if stopc < 1e-6 or k > maxIter:
            converged = True
    
    return v, s
        
        
    
    
        