# https://github.com/pytorch/examples/issues/116

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm

from vgan import VanillaDiscriminator, VanillaGenerator


# function to train the discriminator network
def _train_discriminator(discriminator, optimizer, criterion, data_real, data_fake, real_label, fake_label):
    optimizer.zero_grad()
    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)
    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)
    loss_real.backward()
    loss_fake.backward() # Release the computational graph! retain_graph=False
    optimizer.step() # Only update parameter in D
    return loss_real + loss_fake

# function to train the generator network
def _train_generator(discriminator, optimizer, criterion, data_fake, real_label):
    optimizer.zero_grad()
    output = discriminator(data_fake)
    loss = criterion(output, real_label) # loss occurs when output by fake is different from real by Goodfellow
    loss.backward() # Redundant calculation on D
    optimizer.step()
    return loss

def train_agda(dataset,g_cpu,options):
    nz = options['nz']
    batch_size = options['batch_size']
    num_epochs = options['num_epochs']
    device = options['device']

    # networks
    generator = VanillaGenerator(nz).to(device)
    discriminator = VanillaDiscriminator().to(device)

    print('##### GENERATOR #####')
    print(generator)
    print('######################')
    print('\n##### DISCRIMINATOR #####')
    print(discriminator)
    print('######################')

    # optimizers
    optim_g = optim.SGD(generator.parameters(), lr=0.0002)
    optim_d = optim.SGD(discriminator.parameters(), lr=0.0002)  # only takes in D's parameter

    # loss function
    criterion = nn.BCELoss()

    # Random selection dataloader
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset), generator=g_cpu)
    train_loader = DataLoader(dataset, batch_sampler=BatchSampler(sampler,batch_size=batch_size,drop_last=False))

    # Initialize parameter saving
    gen_param = []
    dis_param = []
    losses_g = []
    losses_d = []

    for epoch in range(num_epochs):
        # Initialize parameter saving for this epoch
        loss_g = 0.0
        loss_d = 0.0
        epoch_gen_param = []
        epoch_dis_param = []

        # batch training
        for i, (images, _, noises) in tqdm(enumerate(train_loader), total=int(len(dataset)/batch_size)): # we don't need the label for imgs
            b_size = images.size()[0]
            images = images.to(device)
            noises = noises.to(device)
            data_fake_detach = generator(noises).detach() # Detach fake from the graph to save computation
            data_real = images
            label_real = torch.ones(b_size, 1).to(device)
            label_fake = torch.zeros(b_size, 1).to(device)
            # train the discriminator network
            loss_d += _train_discriminator(discriminator, optim_d, criterion, data_real, data_fake_detach, label_real, label_fake)
            data_fake = generator(noises) # synchronous noise
            # train the generator network
            loss_g += _train_generator(discriminator, optim_g, criterion, data_fake, label_real)

        # Save parameters
        for param in generator.parameters():
            epoch_gen_param.append(param.data.clone())

        for param in discriminator.parameters():
            epoch_dis_param.append(param.data.clone())

        epoch_loss_g = loss_g / i
        epoch_loss_d = loss_d / i
        losses_g.append(epoch_loss_g)
        losses_d.append(epoch_loss_d)

        gen_param.append(epoch_gen_param)
        dis_param.append(epoch_dis_param)

        print(f"Epoch {epoch+1} of {num_epochs}, Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

    return gen_param, dis_param