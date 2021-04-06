import torch

from custom_data import RDATA
from train import train_agda

def exp_stability(remove_index, manual_seed, options):
    metric = options['metric']
    # Load MNIST dataset and remove one index
    data_1 = RDATA(options['data'], remove_index=remove_index)

    # Training
    gen_param_1, dis_param_1 = train_agda(data_1, manual_seed, options)

    # Load the second dataset
    data_2 = RDATA(options['data'], remove_index=remove_index+1)

    gen_param_2, dis_param_2 = train_agda(data_2, manual_seed, options)

    # Compute the difference
    gen_diff = []
    dis_diff = []
    # parameters by epochs
    for i in range(len(gen_param_1)):
        gen_diff_layer = []
        dis_diff_layer = []
        # parameters in epochs by layers
        for j in range(len(gen_param_1[i])):
            if metric == 'fro':
                gen_distance = torch.norm(gen_param_1[i][j] - gen_param_2[i][j])
                dis_distance = torch.norm(dis_param_1[i][j] - dis_param_2[i][j])
            elif metric == 'ned':
                # https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance
                gen_distance = 0.5 * torch.var(gen_param_1[i][j] - gen_param_2[i][j]) / (torch.var(gen_param_1[i][j]) + torch.var(gen_param_1[i][j]))
                dis_distance = 0.5 * torch.var(dis_param_1[i][j] - dis_param_2[i][j]) / (torch.var(dis_param_1[i][j]) + torch.var(dis_param_1[i][j]))
            else:
                raise RuntimeError('Metric not found')
            gen_diff_layer.append(gen_distance)
            dis_diff_layer.append(dis_distance)
        gen_diff.append(gen_diff_layer)
        dis_diff.append(dis_diff_layer)

    gen_diff = torch.tensor(gen_diff)
    dis_diff = torch.tensor(dis_diff)
    gen_diff = torch.transpose(gen_diff, 0, 1)
    dis_diff = torch.transpose(dis_diff, 0, 1)

    # print('gen_diff')
    # print(gen_diff)

    return gen_diff, dis_diff

if __name__ == '__main__':
    remove_index = 0
    manual_seed = 123
    options = dict()
    options['model'] = 'dcgan'
    options['loss'] = 'wgan'
    options['data'] = 'cifar10'
    options['metric'] = 'fro'
    options['learning_rate'] = 0.0002
    options['nz'] = 8
    options['batch_size'] = 500
    options['num_epochs'] = 2
    options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_stability(remove_index, manual_seed, options)