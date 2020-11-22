import torch
from torch.utils.data import DataLoader, Dataset
'''
Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，
我们再使用DataLoader这个类来更加快捷的对数据进行操作。
DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每
个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据
的时候使用几个子进程)
'''
from torchvision import datasets, transforms
'''
torchvision 主要包含三部分：
models：提供深度学习中各种经典网络的网络结构以及预训练好的模型，包括 
AlexNet 、VGG 系列、ResNet 系列、Inception 系列等；

datasets： 提供常用的数据集加载，设计上都是继承 
torch.utils.data.Dataset，主要包括 MNIST、CIFAR10/100、ImageNet
、COCO等；

transforms：提供常用的数据预处理操作，主要包括对 Tensor 以及 
PIL Image 对象的操作；
'''
import pandas as pd
import numpy as np

import os


def get_data(args):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.root, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])),
        batch_size=args.bs, shuffle=True)
    return train_loader, test_loader

from options import get_args
from util import get_grid

if __name__ == '__main__':
    args = get_args()
    # print(args.bs)
    # print(args.root)
    train_loader, test_loader = get_data(args=args)
    for idx, (x, y) in enumerate(train_loader):
        # print(x)
        # print(x.shape)
        imgs = get_grid(x, args, 1, idx)
        # imgs.show()

