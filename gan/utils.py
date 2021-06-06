import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)
        # torch.nn.init.constant_(m.weight, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)
        # torch.nn.init.constant_(m.weight, 0.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.1)
        # torch.nn.init.constant_(m.weight, 0.0)
        torch.nn.init.zeros_(m.bias)

# def compute_dis_loss(out_real, out_fake, y_real=1, y_fake=0, loss='gan'):
#     if loss == 'gan':
#         dis_loss = - torch.log(out_real + 1e-8).mean() - torch.log(1 - out_fake + 1e-8).mean() # avoid -inf
#     elif loss == 'nsgan':
#         dis_loss = - torch.log(out_real + 1e-8).mean() - torch.log(1 - out_fake + 1e-8).mean()
#     elif loss == 'lsgan':
#         dis_loss = 0.5 * torch.square(out_real - y_real).mean() + 0.5 * torch.square(out_fake - y_fake).mean()
#     elif loss == 'wgan':
#         dis_loss = - out_real.mean() + out_fake.mean()
#     else:
#         raise NotImplementedError()
#
#     return dis_loss
#
# def compute_gen_loss(out_fake, label, loss='gan'):
#     if loss == 'gan':
#         gen_loss = - torch.log(1 - out_fake + 1e-8).mean()
#     elif loss == 'wgan':
#         gen_loss = - out_fake.mean()
#     else:
#         raise NotImplementedError()
#
#     return gen_loss

def compute_gan_loss(output, label, loss='gan'):
    if loss == 'bce':
        gan_loss = nn.BCELoss()(output, label)
    elif loss == 'gan':
        gan_loss = - torch.mean(label * torch.log(output + 1e-8) + (1 - label) * torch.log(1 - output + 1e-8))
    elif loss == 'wgan':
        label = 2 * label - 1  # convert to 1, -1
        gan_loss = torch.mean(label * output)
    else:
        raise NotImplementedError()

    return gan_loss