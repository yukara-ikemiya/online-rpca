# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:44:23 2016

@author: wexiao
"""
from rpca.pcp import pcp
from rpca.utility import solve_proj2
import numpy as np

def stoc_rpca(M, burnin,  lambda1=np.nan, lambda2=np.nan):
    """ 
    Online Robust PCA via Stochastic Optimizaton (Feng, Xu and Yan, 2013)
 
    The loss function is 
        min_{L,S} { 1/2||M-L-S||_F^2 + lambda1||L||_* + lambda2*||S(:)||_1}
    
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
    
    lambda1, lambda2:tuning parameters
    
    burnin : burnin sample size.
    
    Returns
    ----------
    Lhat : array-like, low-rank matrix.
    
    Shat : array-like, sparse matrix.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Feng, Jiashi, Huan Xu, and Shuicheng Yan. 
    Online robust pca via stochastic optimization. Advances in Neural Information Processing Systems. 2013.

    Rule of thumb for tuning paramters:
    lambda1 = 1.0/np.sqrt(max(M.shape));
    lambda2 = lambda1;
    
    """
    m, n = M.shape
    # calculate pcp on burnin samples and find rank r
    Lhat, Shat, niter, r = pcp(M[:,:burnin])
    Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat)
    if np.isnan(lambda1):
        lambda1 = 1.0/np.sqrt(m)/np.mean(sigmas_hat[:r])
    if np.isnan(lambda2):
        lambda2 = 1.0/np.sqrt(m)  
    
    # initialization
    U = np.random.rand(m, r)
#    Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat)
#    U = Uhat[:,:r].dot(np.sqrt(np.diag(sigmas_hat[:r])))
    A = np.zeros((r, r))
    B = np.zeros((m, r))
    for i in range(burnin, n):
        mi = M[:, i]
        vi, si = solve_proj2(mi, U, lambda1, lambda2)
        Shat = np.hstack((Shat, si.reshape(m,1)))
        A = A + np.outer(vi, vi)
        B = B + np.outer(mi - si, vi)
        U = update_col(U, A, B, lambda1)
        Lhat = np.hstack((Lhat, U.dot(vi).reshape(m,1)))
        #Lhat = np.hstack((Lhat, (mi - si).reshape(m,1)))
        
    return Lhat, Shat, r, U
    
def update_col(U, A, B, lambda1):
    m, r = U.shape
    A = A + lambda1*np.identity(r)
    for j in range(r):
        bj = B[:,j]
        uj = U[:,j]
        aj = A[:,j]
        temp = (bj - U.dot(aj))/A[j,j] + uj
        U[:,j] = temp/max(np.linalg.norm(temp), 1)
    
    return U