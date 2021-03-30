import torch

from custom_data import RMNIST
from train import train_agda

def exp_stability(fix_seed, remove_index, generator_seed, options):
    # Load MNIST dataset and remove one index
    mnist_data_1 = RMNIST(remove_index=remove_index, nz=options['nz'], fix_seed=fix_seed)

    # Fix init random generator
    g_cpu = torch.Generator()
    g_cpu.manual_seed(generator_seed)

    # Training
    gen_param_1, dis_param_1 = train_agda(mnist_data_1,g_cpu,options)

    # Load the second dataset
    mnist_data_2 = RMNIST(remove_index=remove_index+1,nz=options['nz'], fix_seed=fix_seed)

    # Fix init random generator again
    g_cpu = torch.Generator()
    g_cpu.manual_seed(generator_seed)

    gen_param_2, dis_param_2 = train_agda(mnist_data_2,g_cpu,options)

    # Compute the difference
    gen_diff = []
    dis_diff = []
    for i in range(len(gen_param_1)):
        gen_diff_layer = []
        dis_diff_layer = []
        for j in range(len(gen_param_1[i])):
            gen_diff_layer.append(torch.norm(gen_param_1[i][j] - gen_param_2[i][j], p='fro'))
            dis_diff_layer.append(torch.norm(dis_param_1[i][j] - dis_param_2[i][j], p='fro'))
        gen_diff.append(gen_diff_layer)
        dis_diff.append(dis_diff_layer)
    gen_diff = torch.tensor(gen_diff)
    dis_diff = torch.tensor(dis_diff)
    gen_diff = torch.transpose(gen_diff, 0, 1)
    dis_diff = torch.transpose(dis_diff, 0, 1)
    return gen_diff, dis_diff