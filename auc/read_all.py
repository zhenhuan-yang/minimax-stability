# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:49:59 2020

@author: Yunwen Lei
"""

import os
import numpy as np


# good mnist usps w8a vowel satimage  ijcnn1 malware acoustic connect covtype webspam_u malware
# good: australia, splice, w8a, satimage, ijcnn, phishing, shuttle, smartbuilding, pendigits
# rcv1


def read_all():
    # print('name is ' + name)
    import matplotlib.pyplot as plt
    # Using seaborn's style
    plt.style.use('seaborn')
    width = 345

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 16,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }

    plt.rcParams.update(tex_fonts)
    # plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05, 0.5))
    plt.figure(figsize=(18,6))
    plt.tight_layout()
    plt.subplot(121)

    for (eta_p, cor, line) in list(
            zip([0.1, 0.3, 1, 3], ['navy', 'maroon', 'orange', 'yellowgreen'], ['--o', '-->', '--*', '--s'])):
        path = 'res/' + 'svmguide3' + str(int(eta_p * 100)) + '.npy'
        if os.path.isfile(path):
            data = np.load(path, allow_pickle=True).item()
            #           options = name['options']
            n_tr = data['n_tr']
            res_idx_1 = np.array(data['res_idx']) / n_tr

            dist_diff_1 = data['dist_diff']
            gen_diff_1 = data['gen_diff']
            plt.errorbar(res_idx_1, dist_diff_1['mean'], yerr=dist_diff_1['std'] * 0.3, color=cor, linewidth=1.5,
                         fmt=line, capsize=3, elinewidth=1, markeredgewidth=1, markersize=5,
                         label=r"$\eta$=" + str(eta_p))  # plt. plo-t   plt.semilog-x

    plt.xlabel('Number of Passes')
    plt.ylabel('Euclidean Distance')
    plt.legend(loc='upper left')
    # plt.title(r'\texttt{svmguide3}')

    plt.subplots_adjust(wspace=.15)

    plt.subplot(122)
    for (eta_p, cor, line) in list(
            zip([0.1, 0.3, 1, 3], ['navy', 'maroon', 'orange', 'yellowgreen'], ['--o', '-->', '--*', '--s'])):
        path = 'res/' + 'w5a' + str(int(eta_p * 100)) + '.npy'
        if os.path.isfile(path):
            data = np.load(path, allow_pickle=True).item()
            #           options = name['options']
            n_tr = data['n_tr']
            res_idx_1 = np.array(data['res_idx']) / n_tr

            dist_diff_1 = data['dist_diff']
            gen_diff_1 = data['gen_diff']
            plt.errorbar(res_idx_1, dist_diff_1['mean'], yerr=dist_diff_1['std'] * 0.3, color=cor, linewidth=1.5,
                         fmt=line, capsize=3, elinewidth=1, markeredgewidth=1, markersize=5,
                         label=r"$\eta$=" + str(eta_p))  # plt. plo-t   plt.semilog-x

    plt.xlabel('Number of Passes')
    plt.legend(loc='upper left')
    # plt.title(r'\texttt{w5a}')
    plt.savefig('res/' + 'all' + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()

if __name__ == '__main__':
    read_all()

