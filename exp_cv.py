import torch
from exp_stability import exp_stability
import os
from itertools import product
import pickle as pkl

data_path = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_path):
    os.mkdir(data_path)

# learning parameters
options = dict()
options['batch_size'] = 128 # large batch size for GPU can speed up
options['num_epochs'] = 10
options['nz'] = 128 # latent vector size
options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: %s' %options['device'])

# data parameters
fix_seed_list = [1, 123, 58317, 962548, 20457563]
remove_index_list = [7, 48, 321, 4592, 26195]
generator_seed_list = [3, 864, 77777, 9345634, 2145483647]

# initialize saving
num_cv = len(fix_seed_list) * len(remove_index_list) * len(generator_seed_list)
gen_diff_cv = torch.zeros(num_cv, 8, options['num_epochs'])
dis_diff_cv = torch.zeros(num_cv, 8, options['num_epochs'])

# cross validation
for i, (fix_seed, remove_index, generator_seed) in enumerate(product(fix_seed_list, remove_index_list, generator_seed_list)):
    print('Current cross validation number: %d / Total: %d' %(i+1, num_cv))
    gen_diff, dis_diff = exp_stability(fix_seed, remove_index, generator_seed, options)
    gen_diff_cv[i] = gen_diff
    dis_diff_cv[i] = dis_diff

results_gen = dict()
results_dis = dict()
results_gen['mean'] = torch.mean(gen_diff_cv,0)
results_gen['std'] = torch.std(gen_diff_cv,0)
results_dis['mean'] = torch.mean(dis_diff_cv,0)
results_dis['std'] = torch.std(dis_diff_cv,0)
results = dict()
results['gen'] = results_gen
results['dis'] = results_dis
results['options'] = options


res_path = os.path.join(os.getcwd(), 'res')
if not os.path.exists(res_path):
    os.mkdir(res_path)
f = open(os.path.join(res_path, 'mnist_%s.pkl' % options['device']), 'wb')
pkl.dump(results, f)
f.close()