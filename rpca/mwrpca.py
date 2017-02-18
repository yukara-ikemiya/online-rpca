# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 11:33:03 2016

@author: wexiao
"""
from rpca.pcp import pcp
import numpy as np

def mwrpca(M, burnin, win_size):
    """ 
    moving window RPCA (Lee and Lee, 2009)
  
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
        
    burnin : burnin sample size.
    
    win_size : length of moving window.
    
    Returns
    ----------
    Lhat : array-like, low-rank matrix.
    
    Shat : array-like, sparse matrix.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Lee, HyeungIll, and Lee, JungWoo. 
    Online update techniques for projection based Robust Principal Component Analysis. ICT Express (2015).
    
    """
    m, n = M.shape
    Lhat, Shat, niter, rank = pcp(M[:,:burnin])
    for i in range(burnin, n):
        dL, dS, niter, rank = pcp(M[:, max(0, i-win_size+1):(i+1)])
        Lhat = np.hstack((Lhat, dL[:,-1].reshape(m,1)))
        Shat = np.hstack((Shat, dS[:,-1].reshape(m,1)))
    
    return Lhat, Shat, rank