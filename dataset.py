# -*- coding: utf-8 -*-
"""
@Author : kk
@Date : 2026/9/21 14:51
@Describe : 数据集加载
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import BATCH_SIZE, DATA_ROOT, IMG_SIZE

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader
