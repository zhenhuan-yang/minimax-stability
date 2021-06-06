import os
import pickle as pkl
import torch

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Plotting')
parser.add_argument('--model', default='vgan', choices={'vgan', 'dcgan'}, help='GAN structure')
# nsgan is non-saturating gan
parser.add_argument('--loss', default='bce', choices={'bce', 'gan', 'wgan'}, help='GAN loss')
parser.add_argument('--data', default='mnist', choices={'mnist', 'cifar10'}, help='Dataset')
parser.add_argument('--metric', default='fro', choices={'fro', 'ned'}, help='Distance metric')
args = parser.parse_args()

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
f = open(os.path.join(res_path, '%s_%s_%s_%s.pkl' % (args.model, args.loss, args.data, args.metric)), 'rb')
results = pkl.load(f)
epochs = torch.arange(results['options']['num_epochs']) + 1

colors = ['orchid', 'darkorange', 'r', 'b']

plt.subplot(121)
for i, color in enumerate(colors):
    plt.plot(epochs, results['gen']['mean'][i],
             linewidth=1.5, linestyle='--', marker='o', markersize=8, color=color, label='layer %d' % (i + 1))
total_mean = torch.sum(results['gen']['mean'], 0) / len(colors)
total_std = torch.sum(results['gen']['std'], 0) / len(colors)
plt.errorbar(epochs, total_mean, yerr=total_std, color='yellowgreen', linestyle='--', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='o', label='total')
plt.fill_between(epochs, total_mean - total_std, total_mean + total_std, facecolor='powderblue')
plt.legend(loc='upper left')
plt.xlabel('Number of Passes')
plt.ylabel('Euclidean Distance')
# plt.title('Generator')

plt.subplot(122)
for i, color in enumerate(colors):
    plt.plot(epochs, results['dis']['mean'][i],
             linewidth=1.5, linestyle='--', marker='o', markersize=8, color=color, label='layer %d' % (i + 1))
total_mean = torch.sum(results['dis']['mean'], 0) / len(colors)
total_std = torch.sum(results['dis']['std'], 0) / len(colors)
plt.errorbar(epochs, total_mean, yerr=total_std, color='yellowgreen', linestyle='--', linewidth=1.5, capsize=4,
             markeredgewidth=1, elinewidth=2, marker='o', label='total')
plt.fill_between(epochs, total_mean - total_std, total_mean + total_std, facecolor='powderblue')
plt.legend(loc='upper left')
plt.xlabel('Number of Passes')
# plt.title('Discriminator')

plt.savefig(os.path.join(res_path, '%s_%s_%s_%s.png' % (args.model, args.loss, args.data, args.metric)),
            dpi=600, bbox_inches='tight', pad_inches=0.05)

plt.show()