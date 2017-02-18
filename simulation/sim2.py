# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:23:42 2016

@author: wexiao
"""
import numpy as np
from timeit import default_timer as timer
import math
import os, sys
sys.path.insert(0, os.path.abspath('..'))

#from rpca.pcp import pcp
from rpca.mwrpca import mwrpca
from rpca.stoc_rpca  import stoc_rpca
from rpca.omwrpca  import omwrpca
from rpca.omwrpca_cp  import omwrpca_cp
from criteria import evaluate
   
def simulation(method, m=400, n=5000, r=10, rho=0.1, n_p=250, r0=5, nrep=50, seed=1234, 
               burnin=200, win_size=200, track_cp_burnin=100, n_check_cp=20, alpha=0.01, proportion=0.5, n_positive=3, min_test_size=100, tolerance_num=0, factor=1):
    result = [np.nan] * nrep
    run_times = [np.nan] * nrep
    random_state = np.random.get_state()
    
    for rep in range(nrep):
        if rep == 0:
            # seed the generator
            np.random.seed(seed=seed)
        else:
            np.random.set_state(random_state)                
        result[rep] = {}
        run_times[rep] = {}
        
        #simulate the data
        U0 = np.random.randn(m,r)
        Utilde = []
        K = int(math.ceil(float(n)/n_p))    
        for k in range(K):
            Utemp = np.random.randn(m,r)
            Utemp[:,r0:] = 0
            Utilde.append(Utemp)
    #    U0 = scipy.linalg.orth(U0)
        V0 = np.random.randn(n + burnin,r)
        L0 = U0.dot(V0[:burnin,:].transpose())
        U = U0
        for k in range(n):
            i = k // n_p
            U += 1.0/n_p*Utilde[i]
            L0 = np.hstack((L0, U.dot(V0[burnin+k, :]).reshape(m,1)))
        S0 = (np.random.uniform(0, 1, size=(m,n + burnin)) < rho).astype(int) * np.random.uniform(-1000, 1000, size=(m,n + burnin))
        M0 = L0+S0
        # save random generator state            
        random_state = np.random.get_state()
        
        if method == "online":
            #stoc_RPCA
            start = timer()
            Lhat, Shat, rank, Uhat = stoc_rpca(M0, burnin, lambda1=1.0/np.sqrt(m), lambda2=1.0/np.sqrt(m)*(10**2))
            end = timer()       
            result[rep]['stoc_rpca'] = evaluate(Lhat, Shat, L0, S0, r, U0, burnin)
            run_times[rep]['stoc_rpca'] = end - start
     
            #omwrpca 
            start = timer()
            Lhat, Shat, rank = omwrpca(M0, burnin, win_size, lambda1=1.0/np.sqrt(m), lambda2=1.0/np.sqrt(m)*(10**2))
            end = timer()
            result[rep]['omwrpca'] = evaluate(Lhat, Shat, L0, S0, r, U0, burnin)
            run_times[rep]['omwrpca'] = end - start
            
            #omwrpca_cp
            start = timer()
            Lhat, Shat, rank, cp, num_sparses = omwrpca_cp(M0, burnin, win_size, track_cp_burnin, n_check_cp, alpha, proportion, n_positive, min_test_size, 
                                                tolerance_num=tolerance_num, lambda1=1.0/np.sqrt(m), lambda2=1.0/np.sqrt(m)*(10**2), factor=factor)
            end = timer()
            result[rep]['omwrpca_cp'] = evaluate(Lhat, Shat, L0, S0, r, U0, burnin)
            result[rep]['omwrpca_cp: rank'] = rank
            result[rep]['omwrpca_cp: cp'] = cp
            run_times[rep]['omwrpca_cp'] = end - start
        elif method == "batch":
            #batch RPCA
#            start = timer()
#            Lhat, Shat, niter, rank = pcp(M0, lam=1/np.sqrt(max(m,burnin)), mu=0.25/np.abs(M0[:,:burnin]).mean())
#            end = timer()
#            result[rep]['batchRPCA'] = {}    
#            error_L = np.linalg.norm(Lhat - L0, 'fro')/np.linalg.norm(L0, 'fro')
#            error_S = np.linalg.norm(Shat - S0, 'fro')/np.linalg.norm(S0, 'fro')
#            false_S = ((S0 != 0) != (Shat != 0)).sum().astype(float)
#            result[rep]['batchRPCA']['error_L'] = error_L
#            result[rep]['batchRPCA']['error_S'] = error_S
#            result[rep]['batchRPCA']['false_S'] = false_S
#            run_times[rep]['batchRPCA'] = end - start
            
            # moving window RPCA (mwrpca)
            start = timer()
            Lhat, Shat, rank = mwrpca(M0, burnin, win_size)
            end = timer()    
            result[rep]['mwrpca'] = evaluate(Lhat, Shat, L0, S0, r, U0, burnin)
            run_times[rep]['mwrpca'] = end - start

    # combine the result    
    sim_result = {}
    sim_result['result'] = result
    sim_result['run_times'] = run_times
    
    return sim_result