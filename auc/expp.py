# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:53:20 2020

@author: Yunwen
w1a 2,477, 300
a1a, 1605, 123
ionosphere, 351, 34
breast, 683, 10
svmguide3, 1243, 21
rcv    n_tr=20,242, d = 47,236
gisette  n_tr = 6000, d = 5,000
adult9 n_tr = 32,561 , d = 123
adult6 n_tr = 1605, d = 123
madelon n_tr = 2,000, d = 500
ijcnn n_tr = 49,990, d = 22
sensorless 58,509, d = 48
splice n_tr = 1000, d = 60
adult5 n_tr = 6,414, d = 123
cod n_tr =  59,535, d = 8
svmguide n_tr = 3,089, d = 4
covtype n_all =  581,012, d = 54
phishing n_all= 11050, d = 68
mushroom n_all= 8124, d = 112
skin n_all = 245,057, d = 3
real-sim 72,309, 20,958

small
diabetes 768, 8     fourclass 862, 2 
splice 3175, 60   usps, 9298, 256,  a9a  32561, 123
mnist 60,000, 780 German 1000, 24
vowel 528, 10     satimage 4,435, 36

huge size
poker 25,010, 10
higgs 11,000,000, 28
acoustic 78,823, 50
connect 67,557, 126
covtype 581,012, 54
IJCNN 141,691, 22
w8a 49,749, 300   letter 15,000, 16
webspam_u  350,000  254
rcv    n_tr=20,242, d = 47,236
gisette  n_tr = 6000, d = 5,000
adult9 n_tr = 32,561 , d = 123
adult6 n_tr = 1605, d = 123
madelon n_tr = 2,000, d = 500
ijcnn n_tr = 49,990, d = 22
splice n_tr = 1000, d = 60
adult5 n_tr = 6,414, d = 123
cod n_tr =  59,535, d = 8
plmguide n_tr = 3,089, d = 4
covtype n_all =  581,012, d = 54
phishing n_all= 11050, d = 68
mushroom n_all= 8124, d = 112
skin n_all = 245,057, d = 3
real-sim 72,309, 20,958

Good: sonar, australia splice, malware, smartBuilding, shuttle, cover, phishing, satimage, w8a, usps, splice-sustech

good: australia, splice, w8a, satimage, ijcnn, phishing, shuttle, smartbuilding, pendigits-germany server



australia, splice, usps, mnist, w8a, phishing, mushroom, shuttle, webspam_u
"""


from exp_stability import exp_stability

from read_auc import read_auc

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stability measured by Euclidean distance')
    parser.add_argument('-d', '--data', default='diabetes', help='name of the dataset (default: diabetes)')
    args = parser.parse_args()

#    for eta_p in [0.1, 0.3, 1, 3]:
#        exp_stability(args.data, eta_p)

    read_auc(args.data)