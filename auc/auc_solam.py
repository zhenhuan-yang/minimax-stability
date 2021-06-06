# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:25:13 2018

@author: Yunwen

# -*- coding: utf-8 -*-
Spyder Editor

We apply the algorithm in Ying, 2016 NIPS to do AUC maximization

Input:
    x_tr: training instances
    y_tr: training labels
    x_te: testing instances
    y_te: testing labels
    options: a dictionary 
        'ids' stores the indices of examples traversed, ids divided by number of training examples is the number of passes
        'eta' stores the initial step size
        'beta': the parameter R or the L-2 regularizer (depending on the algorithm)
        'fast': 1/sqrt(t) or 1/t step
        'n_pass': the number of passes
        'time_lim': optional argument, the maximal time allowed
Output:
    aucs: results on iterates indexed by res_idx
    time:
        
auc_solam_L2 is the original implementation in Ying 2016, NIPS        
auc_solam_L2: is the variant with stochastic proximal AUC maximization, L2 regularizer is processed analogously by natole 2018 ICML
"""
#https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r singleton 
import numpy as np
from sklearn import metrics
from scipy.sparse import isspmatrix




# for this algorithm, beta is the L-2 parameter and the algorithm is the stochastic proximal AUC maximization with the L-2 regularizer
def auc_solam(x_tr, y_tr, x_te, y_te, options):
    # options
    ids = options['ids']
    # eta_0 = options['eta']
    beta = options['beta']  # beta is the parameter R, we use beta for consistency
    # T = len(ids)
    etas = options['etas']
    # print(etas)
    # print(beta)
    res_idx = options['res_idx']
    # series = np.arange(1,T+1,1)
    # if options['fast']:          
    #     etas = 2 / (series * eta_0 + 1)
    # else:
    #     etas = eta_0 / (np.sqrt(series))
    n_tr, dim = x_tr.shape
    v = np.zeros(dim + 2)
    alpha = 0
    sp = 0   # the estimate of probability with positive example
    t = 0    # the time iterate"  
    #-------------------------------
    # for storing the results
    n_idx = len(res_idx)
    ws = np.zeros((n_idx, dim + 3))
    gens = np.zeros(n_idx)
    i_res = 0
    #------------------------------
    gd = np.zeros(dim + 2)
    if isspmatrix(x_tr):
        x_tr = x_tr.toarray()  # to dense matrix for speeding up the computation
    while t < len(ids):
        #print(ids[t])
        x_t = x_tr[ids[t], :]
        y_t = y_tr[ids[t]]
        wx = np.inner(v[:dim], x_t)#np.inner(x_t, v[:dim])
        eta = etas[t]
        t = t + 1
        if y_t == 1:
            sp = sp + 1
            p = sp / t
            gd[:dim] = (1 - p) * (wx - v[dim] - 1 - alpha) * x_t
            gd[dim] = (p - 1) * (wx - v[dim])
            gd[dim+1] = 0
            gd_alpha = (p - 1) * (wx + p * alpha)
        else:
            p = sp / t
            gd[:dim] = p * (wx - v[dim + 1] + 1 + alpha) * x_t
            gd[dim] = 0
            gd[dim+1] = p * (v[dim + 1] - wx)
            gd_alpha = p * (wx + (p - 1) * alpha)
        # print(eta)
        v = v - eta * gd
        alpha = alpha + eta * gd_alpha        
        
        v[:dim] = 1 / (1 + beta * eta) * v[:dim]
        w_ = v[:dim] 
        if i_res < n_idx and res_idx[i_res] == t:                    
            if not np.all(np.isfinite(w_)):
                gens[i_res:] = gens[i_res - 1]    
                ws[i_res:, :] = v
                break
            pred = (x_te.dot(w_.T)).ravel()
            fpr, tpr, thresholds = metrics.roc_curve(y_te, pred.T, pos_label = 1)               
            test_err = metrics.auc(fpr, tpr)
            pred = (x_tr.dot(w_.T)).ravel()
            fpr, tpr, thresholds = metrics.roc_curve(y_tr, pred.T, pos_label = 1)               
            train_err = metrics.auc(fpr, tpr)   
            gens[i_res] = test_err - train_err
            ws[i_res, :dim + 2] = v
            ws[i_res, dim + 2] = alpha
            i_res = i_res + 1
    return ws, gens