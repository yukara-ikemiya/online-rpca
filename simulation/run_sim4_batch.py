# -*- coding: utf-8 -*-
"""
Created on Thu May 05 16:21:40 2016

@author: wexiao
"""
from sim4 import simulation
import cPickle as pickle

# define parameters
pms = {}
pms['method'] = 'batch'

pms['m'] = 400
pms['n'] = 3000
pms['rvec'] = [[10,10,10], [50,50,50], [10,50,25]]
pms['rho'] = [0.01, 0.1]
pms['n_p'] = 250
pms['r0'] = 5
pms['nrep'] = 1
pms['seed'] = 12345

pms['burnin'] = 200
pms['win_size'] = 200

all_results = {}
for i in range(len(pms['rvec'])):
	for j in range(len(pms['rho'])):
		all_results['r%d_rho%d' % (i+1, j+1)] = simulation(method=pms['method'], m=pms['m'], n=pms['n'], 
			rvec=pms['rvec'][i], rho=pms['rho'][j], n_p=pms['n_p'], r0=pms['r0'], nrep=pms['nrep'], seed=pms['seed'], 
		    burnin=pms['burnin'], win_size=pms['win_size'])

all_results['pms'] = pms

with open('result/all_results_sim4_batch.pickle', 'wb') as handle:
    pickle.dump(all_results, handle)