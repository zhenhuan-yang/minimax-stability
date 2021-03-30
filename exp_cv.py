import torch
from exp_stability import exp_stability

# learning parameters
options = dict()
options['batch_size'] = 128 # large batch size for GPU can speed up
options['num_epochs'] = 1
options['nz'] = 128 # latent vector size
options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: %s' %options['device'])

# data parameters
fix_seed = 0
remove_index = 0
generator_seed = 2145483647

# exp_stability(fix_seed, remove_index, generator_seed, options)