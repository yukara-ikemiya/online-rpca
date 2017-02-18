# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 14:25:59 2017

@author: wexiao
"""

import numpy as np
from rpca.utility import thres

def spca2(M, lam=np.nan, mu=np.nan, eta=np.nan, mu0_factor=np.nan, tol=10**(-7), maxit=1000):
    """ Stable Principal Component Pursuit (Zhou et al., 2009)
    
    This code solves the following optimization problem
    min_{L,S} { ||L||_* + lam*||S(:)||_1 + 1/{2*mu}||M-L-S||_F^2}
    using the Accelerated Proximal Gradient method with a decreasing mu_iter.
  
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
        
    lam : positive tuning parameter (default NaN). When lam is set to NaN,  the value 1/sqrt(max(m, n)) * factor 
    will be used in the algorithm, where M is a m by n matrix.
    
    mu: postive tuning parameter (default NaN). When mu is set to NaN, the value sqrt(2*max(m, n)) 
    will be used in the algorithm, where M is a m by n matrix. A good choice of mu is sqrt(2*max(m, n))*sigma,
    where sigma is the standard deviation of error term.
    
    mu0_factor: the initial value of mu_iter is given as min(mu0_factor*mu, 0.99*||M||_F).
    
    eta: each iteration, mu_iter is geomatrically decreasing with the factor eta.
    
    tol : tolerance value for convergency (default 10^-7).
    
    maxit : maximum iteration (default 10000).
    
    Returns
    ----------
    L1 : array-like, low-rank matrix.
    
    S1 : array-like, sparse matrix.
    
    k : number of iteration.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Zhou, Zihan, et al. "Stable principal component pursuit." 
        Information Theory Proceedings (ISIT), 2010 IEEE International Symposium on. IEEE, 2010.
    
    Lin, Zhouchen, et al. "Fast convex optimization algorithms for exact recovery of a corrupted low-rank matrix." 
    Computational Advances in Multi-Sensor Adaptive Processing (CAMSAP) 61.6 (2009).
    
    """
    # parameter setting
    m, n = M.shape
    if np.isnan(mu):
        mu = np.sqrt(2*max(m,n))
    if np.isnan(lam):
        lam = 1.0/np.sqrt(max(m,n))
    if np.isnan(mu0_factor):
        mu0_factor = 1e3
    mu0 = min(mu0_factor*mu, 0.99*np.linalg.norm(M, ord='fro'))   
    if np.isnan(eta):
        eta = 0.9
    
    # initialization
    L0 = np.zeros((m,n)) 
    L1 = np.zeros((m,n)) 
    S0 = np.zeros((m,n))
    S1 = np.zeros((m,n))
    t0 = 1
    t1 = 1
    mu_iter = mu0
    k = 1
    
    while 1:
        Y_L = L1 + (t0-1)/t1*(L1-L0)
        Y_S = S1 + (t0-1)/t1*(S1-S0)
        G_L = Y_L - 0.5*(Y_L + Y_S - M)
        U, sigmas, V = np.linalg.svd(G_L, full_matrices=False);
        rank = (sigmas > mu_iter/2).sum()
        Sigma = np.diag(sigmas[0:rank] - mu_iter/2)
        L0 = L1
        L1 = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
        G_S = Y_S - 0.5*(Y_L + Y_S - M)
        S0 = S1
        S1 = thres(G_S, lam*mu_iter/2)
        t1, t0 = (np.sqrt(t1**2+1) + 1)/2, t1
        mu_iter = max(eta*mu_iter, mu)
        
        # stop the algorithm when converge
        E_L =2*(Y_L - L1) + (L1 + S1 - Y_L - Y_S)
        E_S =2*(Y_S - S1) + (L1 + S1 - Y_L - Y_S) 
        dist = np.sqrt(np.linalg.norm(E_L, ord='fro')**2 + np.linalg.norm(E_S, ord='fro')**2)
        if k >= maxit or dist < tol:
            break
        else:
            k += 1
            
    return L1, S1, k, rank
        

        
    
    
    
    
    
    
    
    
    
    
    
    