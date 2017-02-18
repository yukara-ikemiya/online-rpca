# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:44:23 2016

@author: wexiao
"""
from rpca.pcp import pcp
from rpca.utility import solve_proj2
import numpy as np
from sklearn.utils.extmath import randomized_svd
from collections import deque
import itertools

def omwrpca_cp(M, burnin, win_size, track_cp_burnin, n_check_cp, alpha, proportion, n_positive, min_test_size, 
               tolerance_num=0, lambda1=np.nan, lambda2=np.nan, factor=1):
    """ 
    The loss function is 
        min_{L,S} { 1/2||M-L-S||_F^2 + lambda1||L||_* + lambda2*||S(:)||_1}
    based on moving window.
     
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
    
    burnin : burnin sample size. We require burnin >= win_size.
    
    win_size : length of moving window. We require win_size <= burnin.
    
    track_cp_burnin: the first track_cp_burnin samples generated from omwrpca algorithm will exclude 
    for track change point. Because the result may be unstable.
    
    n_check_cp: buffer size to track changepoint.
    
    alpha: threshold value used in the hypothesis test. Hypothesis test is applied to track subspace changing.
    We suggest use the value 0.01.

    tolerance_num: offset of numbers used in hypothesis test to track change point. A larger tolerance_num gives 
    a more robust result. We restrict tolerance_num to be a non-negative integer. The default value of 
    tolerance_num is 0.
    
    lambda1, lambda2:tuning parameters
    
    factor: parameter factor for PCP.
    
    Returns
    ----------
    Lhat : array-like, low-rank matrix.
    
    Shat : array-like, sparse matrix.
    
    rank : rank of low-rank matrix.
    
    References
    ----------

    Rule of thumb for tuning paramters:
    lambda1 = 1.0/np.sqrt(m);
    lambda2 = 1.0/np.sqrt(m);
    
    """
    m, n = M.shape
    # parameter setting
    assert burnin >= win_size, "Parameter burin should be larger than or equal to parameter win_size."
    if n < burnin:
        print "Parameter burin should be less than or equal to the number of columns of input matrix. Program stops."
        return np.empty((m,0)), np.empty((m,0)), [], [], []
    if np.isnan(lambda1):
        lambda1 = 1.0/np.sqrt(m)
    if np.isnan(lambda2):
        lambda2 = 1.0/np.sqrt(m)
    # calculate pcp on burnin samples and find rank r
    Lhat, Shat, niter, r = pcp(M[:, :burnin], factor=factor)
    
    # initialization for omwrpca
    Uhat, sigmas_hat, Vhat = randomized_svd(Lhat, n_components=r, n_iter=5, random_state=0)
    U = Uhat.dot(np.sqrt(np.diag(sigmas_hat)))
    Vhat_win = Vhat[:, -win_size:]
    A = np.zeros((r, r))
    B = np.zeros((m, r))
    for i in range(Vhat_win.shape[1]):
        A = A + np.outer(Vhat_win[:, i], Vhat_win[:, i])
        B = B + np.outer(M[:, burnin - win_size + i] - Shat[:, burnin - win_size + i], Vhat_win[:, i])
    
    # initialization for change points tracking
    # dist_num_sparses: distribution of the number of nonzero elements of columns of sparse matrix 
    # used for tracking change point
    dist_num_sparses = np.zeros(m+1)
    # buffer_num: number of nonzero elements of columns of sparse matrix in the buffer used for 
    # tracking change point (buffer size = n_check_cp, queue structure)   
    buffer_num  = deque([])
    # buffer_flag: flags of columns of sparse matrix in the buffer used for tracking change point 
    # (buffer size = n_check_cp, queue structure); flag=1 - potential change point; flag=0 - normal point.      
    buffer_flag = deque([])
    # num_sparses, cp, rvec are returned by the function
    # initialize num_sparses to track the number of nonzero elements of columns of sparse matrix
    num_sparses = list((Shat != 0).sum(axis=0))
    # initialize change points to an empty list
    cp = []
    # initialize list of rank to [r]
    rvec = [r]
    
    # main loop
    i = burnin
    while i < n:
        mi = M[:, i]
        vi, si = solve_proj2(mi, U, lambda1, lambda2)
        Shat = np.hstack((Shat, si.reshape(m,1)))
        vi_delete = Vhat_win[:,0]
        Vhat_win = np.hstack((Vhat_win[:,1:], vi.reshape(r,1)))
        A = A + np.outer(vi, vi) - np.outer(vi_delete, vi_delete)
        B = B + np.outer(mi - si, vi) - np.outer(M[:, i - win_size] - Shat[:, i - win_size], vi_delete)
        U = update_col(U, A, B, lambda1)
        Lhat = np.hstack((Lhat, U.dot(vi).reshape(m,1)))
        num_sparses.append((si.reshape(m,1) != 0).sum())
        if i >= burnin + track_cp_burnin and i < burnin + track_cp_burnin + min_test_size:
            num = (si != 0).sum()
            dist_num_sparses[num] += 1
        elif i >= burnin + track_cp_burnin + min_test_size: # do hypothesis testing to find chang point
            num = (si != 0).sum()
            buffer_num.append(num)
            pvalue = dist_num_sparses[max(num - tolerance_num, 0):].sum() / dist_num_sparses.sum()
            if pvalue <= alpha:
                buffer_flag.append(1)
            else:
                buffer_flag.append(0)
            if len(buffer_flag) >= n_check_cp: # check change point
                if len(buffer_flag) == n_check_cp + 1:
                    dist_num_sparses[buffer_num[0]] += 1
                    buffer_num.popleft()
                    buffer_flag.popleft()
                nabnormal = sum(buffer_flag)
                # potential change identified
                if nabnormal >= n_check_cp * float(proportion):
                    for k in range(n_check_cp - n_positive +1):
                        # use the earliest change point if change point exists
                        if sum(itertools.islice(buffer_flag, k, k+n_positive)) == n_positive:
                            changepoint = i - n_check_cp + 1 + k 
                            cp.append(changepoint)
                            Lhat = Lhat[:, :changepoint]
                            Shat = Shat[:, :changepoint]
                            M_update = M[:, changepoint:]
                            num_sparses = num_sparses[:changepoint]
                            # recursively call omwrpca_cp
                            Lhat_update, Shat_update, rvec_update, cp_update, num_sparses_update = \
                            omwrpca_cp(M_update, burnin, win_size, track_cp_burnin, n_check_cp, alpha, 
                                       proportion, n_positive, min_test_size, tolerance_num, lambda1, lambda2, factor)
                            # update Lhat, Shat, rvec, num_sparses, cp
                            Lhat = np.hstack((Lhat, Lhat_update))
                            Shat = np.hstack((Shat, Shat_update))
                            rvec.extend(rvec_update)
                            num_sparses.extend(num_sparses_update)
                            cp.extend([changepoint + j for j in cp_update])
                            return Lhat, Shat, rvec, cp, num_sparses
        i += 1
                    
    return Lhat, Shat, rvec, cp, num_sparses
    
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