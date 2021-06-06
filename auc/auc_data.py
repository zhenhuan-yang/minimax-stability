# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 07:49:34 2021

@author: Yunwen Lei
"""

import numpy as np
from scipy import io as spio
import scipy.sparse as sp
from sklearn import preprocessing 
from sklearn.datasets import load_svmlight_file
import os

def auc_data(data):
    #------------------------  

    #SGD for AUC optimization without regularization

    cur_path = os.getcwd()
    data_path = os.path.join(cur_path,'data')
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #-----------------------------------------
    # processing the data
    print(data)
    if data == 'rcv1' or data == 'gisette' or data == 'ijcnn1' or data == 'madelon' or data == 'svmguide1' or data == 'svmguide3':
        x_tr, y_tr = load_svmlight_file(os.path.join(data_path,data))
        x_te, y_te = load_svmlight_file(os.path.join(data_path,data) + '.t')
#        n_tr, n_te = len(y_tr), len(y_te)
        x = sp.vstack((x_tr, x_te))
        y = np.hstack((y_tr, y_te))
    elif data == 'CCAT' or data == 'astro' or data == 'cov1':
        tmp = spio.loadmat(os.path.join(data_path,data))
        x, y = tmp['Xtrain'], tmp['Xtest']
    elif data == 'protein_h':
        data_tr = np.loadtxt(os.path.join(data_path,data))
#        data_te = np.loadtxt('data/'+data+'_t')
        x, y = data_tr[:,3:], data_tr[:,2]
        x = min_max_scaler.fit_transform(x)
#        x_te, y_te = data_te[:,3:], data_te[:,2]
#        x = sp.vstack((x_tr, x_te))
#        y = np.hstack((y_tr, y_te))
    elif data == 'smartBuilding' or data == 'malware':
        data_ = spio.loadmat(os.path.join(data_path,data))['data']
        x = data_[:, 1:]
        x = min_max_scaler.fit_transform(x)
        y = data_[:, 0]
    elif data == 'http' or data == 'smtp' or data == 'shuttle' or data == 'cover':
        tmp = spio.loadmat(os.path.join(data_path,data))
        x = tmp['X']
        y = tmp['y'].ravel()
        y = np.int16(y)
        x = min_max_scaler.fit_transform(x)
        tmp = []        
    else:
        x, y = load_svmlight_file(os.path.join(data_path,data))
        x = x.toarray()
    n_data, n_dim = x.shape  
    print('(n,dim)=(%s,%s)' % (n_data, n_dim))
    x = preprocessing.normalize(x)  #normalization
    # check the categories of the data, if it is the multi-class data, process it to binary-class
    uLabel = np.unique(y)
    print('uLabel = %s' % uLabel)
    uNum = len(uLabel)    
    if uNum == 2:
        y[y!=1] = -1
    if uNum > 2:
#        uSort = np.random.permutation(uNum)
#        uSort = np.arange(uNum)
        ty = y
        y = np.ones(n_data, dtype=int)
        for k in np.arange(int(uNum / 2), uNum, dtype=int): #negative class
#            print(uLabel[k])
            y[ty == uLabel[k]] = -1
    
    return x, y
    # end of processing the data
    #------------------------------------
    
"""
This function calculates the indices for storing the results

Input:
    n_data: number of training examples
    n_pass: number of passes
    
Output:
    idx: the indices
"""
def get_res_idx(n_iter, options):
    if options['log_res']: 
        res_idx = 2 ** (np.arange(4, np.log2(n_iter), options['rec_log']))
    else:
        res_idx = np.arange(1, n_iter, options['rec'])
    res_idx[-1] = n_iter
    res_idx = [int(i) for i in res_idx]#map(int, res_idx)
    return res_idx
    
"""
This function calculates the indices for SGD

Input:
    n_data: number of training examples
    n_pass: number of passes
    
Output:
    idx: the indices
"""
def get_idx(n_data, n_pass):
    idx = np.zeros(n_data * n_pass, dtype=int)
    # random permutation
    # for i_pass in np.arange(n_pass):
        # idx[i_pass * n_data : (i_pass + 1) * n_data] = np.random.permutation(n_data)
    # random selection
    for i in range(n_data * n_pass):
        idx[i] = np.random.randint(0,high=n_data)
    return idx
    