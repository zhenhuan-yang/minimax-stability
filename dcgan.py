# DCGAN structure for CIFAR10 [3,32,32]
# Also apply to MNIST [1,28,28] -> [1,32,32] by specifying n_out=1, n_in=1

import torch.nn as nn
import torch

# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

class DCGANGenerator(nn.Module):
    '''
    Noise [bs, nz, 1, 1] -> Fake Image [bs, 3, 32, 32]
    '''
    def __init__(self, n_z, n_out=3, n_filters=32): # 32 is chosen so the output dimension can be matched
        super(DCGANGenerator, self).__init__()

        self.n_z = n_z
        self.n_out = n_out
        self.n_filters = n_filters
        # [bs, nz, 1, 1] -> [bs, 128, 4, 4]
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, self.n_filters * 4, 4, bias=False),
            # nn.BatchNorm2d(n_filters * 4),
            nn.ReLU())
        # [bs, 128, 4, 4] -> [bs, 64, 8, 8]
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.n_filters * 4, self.n_filters * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters * 2),
            nn.ReLU())
        # [bs, 64, 8, 8] -> [bs, 32, 16, 16]
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.n_filters * 2, self.n_filters, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters),
            nn.ReLU())
        # [bs, 32, 16, 16] -> [bs, 3, 32, 32]
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.n_filters, self.n_out, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        x = x.view(-1, self.n_z, 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        return x

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

class DCGANDiscriminator(nn.Module):
    '''
    Image [bs, 3, 32, 32] -> [bs]
    '''
    def __init__(self, n_in=3, n_filters=32):
        super(DCGANDiscriminator, self).__init__()

        self.n_in = n_in
        self.n_filters = n_filters
        # [bs, 3, 32, 32] -> [bs, 32, 16, 16]
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_in, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2))
        # [bs, 32, 16, 16] -> [bs, 64, 8, 8]
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2))
        # [bs, 64, 8, 8] -> [bs, 128, 4, 4]
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2))
        # [bs, 128, 4, 4] -> [bs, 1, 1, 1]
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_filters * 4, 1, 4, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 1).squeeze(1)

        return x

if __name__ == '__main__':
    nz = 100
    netG = DCGANGenerator(nz)
    netD = DCGANDiscriminator()
    noise = torch.randn(10, nz)
    datum = torch.randn(10, 3, 32, 32)
    image = netG(noise)
    label = netD(datum)
    print(image.shape)
    print(label.shape)