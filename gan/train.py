# https://github.com/pytorch/examples/issues/116

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm
import random

from utils import weights_init, compute_gan_loss


# function to train the discriminator network
# def _train_discriminator(discriminator, optimizer, data_real, data_fake, loss):
#     optimizer.zero_grad()
#     output_real = discriminator(data_real)
#     output_fake = discriminator(data_fake)
#     loss = compute_dis_loss(output_real, output_fake, loss=loss)
#     loss.backward() # Release the computational graph! retain_graph=False
#     optimizer.step() # Only update parameter in D
#     return loss

# function to train the generator network
# def _train_generator(discriminator, optimizer, data_fake, loss):
#     optimizer.zero_grad()
#     output = discriminator(data_fake)
#     loss = compute_gen_loss(output, loss=loss) # loss occurs when output by fake is different from real
#     loss.backward() # Redundant calculation on D
#     optimizer.step()
#     return loss

def train_agda(dataset, manual_seed, options):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    model = options['model']
    loss = options['loss']
    data = options['data']
    lr = options['learning_rate']
    nz = options['nz']
    batch_size = options['batch_size']
    num_epochs = options['num_epochs']
    device = options['device']

    # Define gan networks
    if model == 'vgan':
        from vgan import VanillaDiscriminator, VanillaGenerator

        if data == 'mnist':
            generator = VanillaGenerator(nz).to(device)
            discriminator = VanillaDiscriminator().to(device)
        elif data == 'cifar10':
            generator = VanillaGenerator(nz, n_c=3).to(device)
            discriminator = VanillaDiscriminator(n_c=3).to(device)
    elif model == 'dcgan':
        from dcgan import DCGANDiscriminator, DCGANGenerator

        if data == 'mnist':
            generator = DCGANGenerator(nz, n_out=1).to(device)
            discriminator = DCGANDiscriminator(n_in=1).to(device)
        elif data == 'cifar10':
            generator = DCGANGenerator(nz).to(device)
            discriminator = DCGANDiscriminator().to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # init_gen_param = 0.0
    # init_dis_param = 0.0
    # for param in generator.parameters():
    #     init_gen_param += torch.norm(param.data.clone())
    # for param in discriminator.parameters():
    #     init_dis_param += torch.norm(param.data.clone())
    #
    # print('generator initial norm: %f' % init_gen_param)
    # print('discriminator initial norm: %f' % init_dis_param)

    # print('##### GENERATOR #####')
    # print(generator)
    # print('######################')
    # print('\n##### DISCRIMINATOR #####')
    # print(discriminator)
    # print('######################')

    # optimizers
    optim_g = optim.SGD(generator.parameters(), lr=lr)
    optim_d = optim.SGD(discriminator.parameters(), lr=lr)  # only takes in D's parameter

    # Initialize parameter saving
    gen_param = []
    dis_param = []

    print('Training......')

    for epoch in range(num_epochs):
        # Random selection dataloader
        sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
        train_loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False))

        # Initialize parameter saving for this epoch
        epoch_gen_param = []
        epoch_dis_param = []

        losses_g = 0.0
        losses_d = 0.0

        # batch training
        # for i, (images, _, noises) in tqdm(enumerate(train_loader), total=int(len(dataset)/batch_size)): # we don't need the label for imgs
        for i, (images, _) in tqdm(enumerate(train_loader, 0), total=int(len(dataset)/batch_size)):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            discriminator.zero_grad()
            images = images.to(device)
            b_size = images.size()[0]
            label = torch.full((b_size,), 1, dtype=images.dtype, device=device)
            output = discriminator(images)
            # loss_real = - torch.mean(1 * torch.log(output + 1e-8))
            loss_real = compute_gan_loss(output, label, loss=loss)
            # loss_real = nn.BCELoss()(output, label)
            loss_real.backward()
            D_x = output.mean().item()

            # train with fake
            noises = torch.randn(b_size, nz, device=device)
            images_fake = generator(noises)
            label.fill_(0)
            output = discriminator(images_fake.detach()) # Detach fake from the graph to save computation
            # loss_fake = - torch.mean(1 * torch.log(1 - output + 1e-8))
            loss_fake = compute_gan_loss(output, label, loss=loss)
            # loss_fake = nn.BCELoss()(output, label)
            loss_fake.backward()
            D_G_z1 = output.mean().item()
            loss_d = loss_real + loss_fake
            optim_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(1)
            output = discriminator(images_fake)
            # loss_g = - torch.mean(1 * torch.log(output + 1e-8))
            loss_g = compute_gan_loss(output, label, loss=loss)
            # loss_g = nn.BCELoss()(output, label)
            loss_g.backward()
            D_G_z2 = output.mean().item()
            optim_g.step()

            # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
            #       % (epoch, num_epochs, i, len(train_loader), loss_d.item(), loss_g.item(), D_x, D_G_z1, D_G_z2))

            losses_d += loss_d.item()
            losses_g += loss_g.item()

        print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, losses_d / i, losses_g / i, D_x, D_G_z1, D_G_z2))

        # Save parameters
        for param in generator.parameters():
            epoch_gen_param.append(param.data.clone()) # When you use .data, you get a new Tensor with requires_grad=False, so cloning it won’t involve autograd

        for param in discriminator.parameters():
            epoch_dis_param.append(param.data.clone())

        # epoch_loss_g = loss_g / i
        # epoch_loss_d = loss_d / i
        # losses_g.append(epoch_loss_g)
        # losses_d.append(epoch_loss_d)

        gen_param.append(epoch_gen_param)
        dis_param.append(epoch_dis_param)

    return gen_param, dis_param

if __name__ == '__main__':
    manual_seed = 123
    options = dict()
    options['model'] = 'dcgan'
    options['loss'] = 'wgan'
    options['data'] = 'cifar10'
    options['metric'] = 'frobenius'
    options['learning_rate'] = 0.0002
    options['nz'] = 8
    options['batch_size'] = 500
    options['num_epochs'] = 2
    options['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if options['data'] == 'mnist':
        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='./data/', download=True, transform=transform)
    elif options['data'] == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.CIFAR10(root='./data/', download=True, transform=transform)
    train_agda(dataset, manual_seed, options)