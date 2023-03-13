from typing import Tuple, Any

from torchvision import datasets, transforms

import utils.IID
from utils.Tools import *
from PIL import Image


class CIFAR10:
    def __init__(self, clients, iid_config):
        transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 获取数据集
        train_datasets = datasets.CIFAR10(root='../data/', train=True,
                                        transform=transformer, download=True)
        test_datasets = datasets.CIFAR10(root='../data/', train=False,
                                        transform=transformer, download=True)
        self.raw_data = train_datasets.data
        self.train_labels = np.array(train_datasets.targets)
        self.test_datasets = test_datasets
        self.train_data = train_datasets.data
        self.test_data = test_datasets.data

        self.train_data_size = self.train_data.shape[0]
        self.datasets = []
        self.iid_config = iid_config

        if isinstance(iid_config, bool):
            print("generating iid data...")
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_labels = self.train_labels[order]
            shard_size = self.train_data_size // clients // 2
            for i in range(clients):
                client_data1 = self.train_data[shard_size * i: shard_size * (i + 1)]
                client_data2 = self.train_data[
                               shard_size * clients + shard_size * i: shard_size * clients + shard_size * (i + 1)]
                client_label1 = self.train_labels[shard_size * i: shard_size * (i + 1)]
                client_label2 = self.train_labels[
                                shard_size * clients + shard_size * i: shard_size * clients + shard_size * (i + 1)]
                client_data, client_label = np.vstack(
                    (client_data1, client_data2)), np.hstack(
                    (client_label1, client_label2))
                self.datasets.append(PicTensorDataset(torch.tensor(client_data), torch.tensor(client_label), transform=transformer))
        else:
            print("generating non_iid data...")
            utils.IID.generate_non_iid_data(iid_config, self, clients, self.train_labels.min(),
                                            self.train_labels.max() + 1, PicTensorDataset, params={'transform': transformer})
        print("data generation process completed")

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.datasets

    def get_config(self):
        return self.iid_config


class PicTensorDataset(TensorDataset):
    def __init__(self, *tensors, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        super(PicTensorDataset, self).__init__(tensors[0], tensors[1])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = tuple(tensor[index] for tensor in self.tensors)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
