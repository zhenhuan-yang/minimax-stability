# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 20:58:42 2018

@author: Yunwen
"""

import numpy as np
from scipy import io as spio
import scipy.sparse as sp
from sklearn.model_selection import RepeatedKFold
from auc_solam import auc_solam
from auc_data import get_idx, get_res_idx
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import itertools
import os
import matplotlib.pyplot as plt
from auc_data import auc_data

def exp_stability(data, eta_p):
    x, y = auc_data(data)
    options = dict()
    options['n_proc'] = 40
    options['n_repeates'] = 10
    options['n_pass'] = 6
    options['data'] = data
    options['n_cv'] = 10
    options['rec_log'] = .5
    options['rec'] = 100
    options['log_res'] = True
    options['beta'] = 0
    options['eta_p'] = eta_p
    n_rep = options['n_repeates']
    n_tr = int(len(y)*0.8)
    n_iter = options['n_pass'] * n_tr
    res_idx = get_res_idx(options['n_pass'] * (n_tr - 1), options)
    options['etas'] = options['eta_p'] * np.ones(n_iter) / np.sqrt(n_iter)
    n_idx = len(res_idx)
    options['res_idx'] = res_idx
    rkf = RepeatedKFold(n_splits=5, n_repeats=n_rep)
    with ProcessPoolExecutor(options['n_proc']) as executor:
        results = executor.map(help_exp, itertools.repeat((x, y, options)), rkf.split(x))
    dist_ret = np.zeros([n_rep * 5, n_idx])
    gen_ret = np.zeros([n_rep * 5, n_idx])
    k = 0
    for ret_dist_, ret_gen_ in results:
        dist_ret[k] = ret_dist_
#        print(auc_ret[k])
        gen_ret[k] = ret_gen_
        k = k + 1  
    dist_diff = dict()
    gen_diff = dict()
    dist_diff['mean'] = np.mean(dist_ret, 0)
    dist_diff['std'] = np.std(dist_ret, 0)
    gen_diff['mean'] = np.mean(gen_ret, 0)
    gen_diff['std'] = np.std(gen_ret, 0)
    np.save('res/' + data + str(options['eta_p'] * 100) + '.npy', {'options':options, 'n_tr':n_tr, 'res_idx':res_idx, 'dist_diff':dist_diff, 'gen_diff':gen_diff}) 
    # plt.plot(res_idx, dist_diff['mean'], color='red', linewidth=2.5, linestyle='-')
    # plt.plot(res_idx, gen_diff['mean'], color='blue', linewidth=2.5, linestyle='--') 
    # plt.savefig('res/' + data  + '.png')
    # plt.show()
    
def help_exp(arg1, arg2):
    arg = arg1 + (arg2,)
    return auc_exp_(*arg)
    
def auc_exp_(x, y, options, idx):
    idx_tr = idx[0]
    idx_te = idx[1]
    x_tr, x_te = x[idx_tr], x[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    
    x_tr_, y_tr_ = x_tr[:-1], y_tr[:-1]
    n_pass = options['n_pass']
    n_tr = len(y_tr_)
    options['ids'] = get_idx(n_tr, n_pass)
    
    ws_a, gens_a = auc_solam(x_tr_, y_tr_, x_te, y_te, options)
    
    # a neighboring dataset
    x_tr_[-1,:] = x_tr[-1, :]
    y_tr_[-1] = y_tr[-1]
    ws_b, gens_b = auc_solam(x_tr_, y_tr_, x_te, y_te, options)
    
    n_res = len(gens_a)
    ret_dist = np.zeros(n_res)
    ret_gen = np.zeros(n_res)
    for i in range(n_res):
        tmp = ws_a[i] - ws_b[i]
        ret_dist[i] = np.linalg.norm(tmp)
    ret_gen = (gens_a + gens_b) / 2
    return ret_dist, ret_gen