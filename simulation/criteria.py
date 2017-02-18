# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 13:32:58 2016

@author: wexiao
"""
import numpy as np
import pandas as pd

#def expressedValue(U1, U2):
#    """
#    calculate the similarity (Expressed Variance E.V.) between subspace U1 and U2
#    """
#    A = np.dot(U2.transpose(), U1)
#    return np.trace(np.dot(A, A.transpose())) / np.trace(np.dot(U1, U1.transpose()))
    
def evaluate(Lhat, Shat, L0, S0, rank, U0, burnin):

    m, n = Lhat.shape
    # initialization    
    output = pd.DataFrame({'t': range(1, n+1-burnin)})  
    error_L = []
    error_S = []
    false_S = []
    sum1_L = 0
    sum2_L = 0
    sum1_S = 0
    sum2_S = 0
    nfalse = 0
    # main loop
    for i in range(burnin, n):
#        Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat[:,:i])
#        EV.append(expressedValue(U0[:,0:rank], Uhat[:,0:rank]))
        # error_L = ||Lhat-L0||_F/||L0||_F
        sum1_L += np.sum(np.square(np.abs(Lhat[:,i] - L0[:,i])))
        sum2_L += np.sum(np.square(np.abs(L0[:,i])))
        error_L.append(np.sqrt(sum1_L)/np.sqrt(sum2_L))
        # error_S = ||Shat-S0||_F/||S0||_F
        sum1_S += np.sum(np.square(np.abs(Shat[:,i] - S0[:,i])))
        sum2_S += np.sum(np.square(np.abs(S0[:,i])))
        error_S.append(np.sqrt(sum1_S)/np.sqrt(sum2_S))
        # number of misclassified entries in Shat 
        nfalse += ((S0[:,i]  != 0) != (Shat[:,i]  != 0)).sum()
        false_S.append(nfalse)
    output['error_L'] = error_L
    output['error_S'] = error_S
    output['false_S'] = false_S
    
    return output