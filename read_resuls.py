import os
import pickle as pkl
import torch

import matplotlib.pyplot as plt

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

plt.figure(figsize=(16, 6))

res_path = os.path.join(os.getcwd(), 'res')
f = open(os.path.join(res_path, 'mnist_cpu.pkl'), 'rb')
results = pkl.load(f)
epochs = torch.arange(results['options']['num_epochs']) + 1

colors = ['orchid', 'darkorange']

plt.subplot(121)
for i, color in enumerate(colors):
    plt.plot(epochs, results['gen']['mean'][i * 2] + results['gen']['mean'][i * 2 + 1],
             linewidth=1.5, linestyle='--', marker='o', markersize=8, color=color, label='layer %d' % (i + 1))
total_mean = torch.sum(results['gen']['mean'], 0) / 2.
total_std = torch.sum(results['gen']['std'], 0) / 2.
plt.errorbar(epochs, total_mean, yerr=total_std, color='yellowgreen', linestyle='--', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='o', label='total')
plt.fill_between(epochs, total_mean - total_std, total_mean + total_std, facecolor='powderblue')
plt.legend(loc='upper left')
plt.xlabel('Number of Passes')
plt.ylabel('Euclidean Distance')
plt.title('Generator')

plt.subplot(122)
for i, color in enumerate(colors):
    plt.plot(epochs, results['dis']['mean'][i * 2] + results['dis']['mean'][i * 2 + 1],
             linewidth=1.5, linestyle='--', marker='o', markersize=8, color=color, label='layer %d' % (i + 1))
total_mean = torch.sum(results['dis']['mean'], 0) / 2.
total_std = torch.sum(results['dis']['std'], 0) / 2.
plt.errorbar(epochs, total_mean, yerr=total_std, color='yellowgreen', linestyle='--', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='o', label='total')
plt.fill_between(epochs, total_mean - total_std, total_mean + total_std, facecolor='powderblue')
plt.legend(loc='upper left')
plt.xlabel('Number of Passes')
plt.title('Discriminator')

plt.savefig(os.path.join(res_path, 'mnist_cpu.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()