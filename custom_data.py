# https://discuss.pytorch.org/t/removing-datapoints-from-dataset/83459

from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch

from typing import Any, Tuple
from PIL import Image

class RDATA(Dataset):
    def __init__(self, dataset, remove_index=0):
        self.dataset = dataset
        if self.dataset == 'mnist':
            self.rawdata = datasets.MNIST(root='./data/', train=True, download=True, transform=None)
        elif self.dataset == 'cifar10':
            self.rawdata = datasets.CIFAR10(root='./data/', train=True, download=True, transform=None)
        else:
            raise RuntimeError('Dataset not found')
        self.rdata, self.rtargets = self.__remove__(remove_index)

    # check the abstract class: mandatory overwrite
    def __getitem__(self, index) -> Tuple[Any, Any]:
        data, target = self.rdata[index], self.rtargets[index]
        if self.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.Resize(32), # to match the dimension of cifar10
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
            data = Image.fromarray(data.numpy(), mode='L') # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
            data = transform(data)
        elif self.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            data = Image.fromarray(data)
            data = transform(data)
        return data, target

    # optional overwrite
    def __len__(self):
        return len(self.rdata)

    def __remove__(self, remove_index):
        # by default we remove the first example
        # when creating the neighborhood dataset remove the second example
        mask = [True] * len(self.rawdata)
        mask[remove_index] = False
        rdata = self.rawdata.data[mask]
        if self.dataset == 'mnist':
            rtargets = self.rawdata.targets[mask]
        elif self.dataset == 'cifar10':
            rtargets = torch.tensor(self.rawdata.targets)[mask]
        return rdata, rtargets



# class RDATA_Deprecated(Dataset):
#     def __init__(self, dataset, remove_index=0, nz=8, fix_seed=0):
#         self.dataset = dataset
#         if self.dataset == 'mnist':
#             self.rawdata = datasets.MNIST(root='./data/', train=True, download=True, transform=None)
#         elif self.dataset == 'cifar10':
#             self.rawdata = datasets.CIFAR10(root='./data/', train=True, download=True, transform=None)
#         else:
#             raise RuntimeError('Dataset not found')
#         self.noises = self.__noise__(nz, fix_seed)
#         self.rdata, self.rtargets, self.rnoises = self.__remove__(remove_index)
#
#     # check the abstract class: mandatory overwrite
#     def __getitem__(self, index) -> Tuple[Any, Any, Any]:
#         data, target, noise = self.rdata[index], self.rtargets[index], self.rnoises[index]
#         if self.dataset == 'mnist':
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
#             data = Image.fromarray(data.numpy(), mode='L') # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
#             data = transform(data)
#         elif self.dataset == 'cifar10':
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
#             data = Image.fromarray(data)
#             data = transform(data)
#         else:
#             raise RuntimeError('Dataset not found')
#         return data, target, noise
#
#     # optional overwrite
#     def __len__(self):
#         return len(self.rdata)
#
#     def __remove__(self, remove_index):
#         # by default we remove the first example
#         # when creating the neighborhood dataset remove the second example
#         mask = [True] * len(self.rawdata)
#         mask[remove_index] = False
#         rdata = self.rawdata.data[mask]
#         if self.dataset == 'mnist':
#             rtargets = self.rawdata.targets[mask]
#         elif self.dataset == 'cifar10':
#             rtargets = torch.tensor(self.rawdata.targets)[mask]
#         rnoises = self.noises[mask]
#         return rdata, rtargets, rnoises
#
#     def __noise__(self, nz, fix_seed):
#         torch.manual_seed(fix_seed)
#         noises = torch.randn(len(self.rawdata), nz)
#         return noises