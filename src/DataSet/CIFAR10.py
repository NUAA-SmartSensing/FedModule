import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

# 数据增广方法
transform = transforms.Compose([
    # +4填充至36x36
    transforms.Pad(4),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机裁剪至32x32
    transforms.RandomCrop(32),
    # 转换至Tensor
    transforms.ToTensor(),
    #  归一化
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
    #                                      std=(0.5, 0.5, 0.5))
])


class CIFAR:
    def __init__(self, clients, is_iid=False):
        # 获取数据集
        train_datasets = datasets.CIFAR10(root='../data/', train=True,
                                          transform=transform, download=True)
        test_datasets = datasets.CIFAR10(root='../data/', train=False,
                                         transform=transform, download=True)
        train_data = train_datasets.data
        self.train_labels = train_datasets.targets
        test_data = test_datasets.data
        self.test_datasets = test_datasets

        self.train_data_size = train_data.shape[0]
        self.datasets = []

        if is_iid:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_labels = self.train_labels[order]
        else:
            # 待更新
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_labels = self.train_labels[order]

        shard_size = self.train_data_size // clients
        for i in range(clients):
            client_data = self.train_data[shard_size * i: shard_size * (i + 1)]
            client_label = self.train_labels[shard_size * i: shard_size * (i + 1)]
            self.datasets.append(TensorDataset(client_data, client_label))

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.datasets
