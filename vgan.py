# Vanilla GAN structure for MNIST generation [1,28,28] -> [1,32,32]
# Also apply to CIFAR10 [3,32,32] by specifying n_c = 3
import torch.nn as nn
import torch

class VanillaGenerator(nn.Module):
    def __init__(self, n_z, n_c=1, n_out=1024):
        super(VanillaGenerator, self).__init__() # to be able to call superclass in subclass

        self.n_z = n_z
        self.n_c = n_c
        self.n_out = n_c * n_out
        self.fc1 = nn.Sequential(
                    nn.Linear(self.n_z, 256, bias=False),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(256, 512, bias=False),
                    nn.LeakyReLU(0.2)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(512, 1024, bias=False),
                    nn.LeakyReLU(0.2)
                    )
        self.fc4 = nn.Sequential(
                    nn.Linear(1024, self.n_out, bias=False),
                    nn.Tanh()
                    )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(-1, self.n_c, 32, 32) # -1 is the dimension of batch size and (1, 32, 32) is the resized dimension
        return x

class VanillaDiscriminator(nn.Module):
    def __init__(self, n_c=1, n_in=1024):
        super(VanillaDiscriminator, self).__init__()

        self.n_c = n_c
        self.n_in = n_c * n_in
        self.fc1 = nn.Sequential(
                    nn.Linear(self.n_in, 1024, bias=False),
                    nn.LeakyReLU(0.2),
                    # nn.Dropout(0.3)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(1024, 512, bias=False),
                    nn.LeakyReLU(0.2),
                    # nn.Dropout(0.3)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(512, 256, bias=False),
                    nn.LeakyReLU(0.2),
                    # nn.Dropout(0.3)
                    )
        self.fc4 = nn.Sequential(
                    nn.Linear(256, 1, bias=False),
                    nn.Sigmoid()
                    )
    def forward(self, x):
        x = x.view(-1, self.n_in) # -1 is the dimension of batch size and 1024 is the dimension of mnist image 1*28*28
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.squeeze(1)
        return x

if __name__ == '__main__':
    nz = 100
    netG = VanillaGenerator(nz)
    netD = VanillaDiscriminator()
    noise = torch.randn(10, nz)
    datum = torch.randn(10, 1, 32, 32)
    image = netG(noise)
    label = netD(datum)
    print(image.shape)
    print(label.shape)