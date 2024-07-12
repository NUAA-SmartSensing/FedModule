import numpy as np
from torchvision import transforms, datasets

from dataset.BaseDataset import BaseDataset
from utils.DatasetUtils import CompositeDataset


class CompositeAbstractDataset(BaseDataset):
    def __init__(self, iid_config):
        super().__init__(iid_config)
        # 获取数据集
        self.train_labels = [None, None]
        self.test_labels = [None, None]
        self.train_data = [None, None]
        self.test_data = [None, None]
        self.train_dataset = None
        self.test_dataset = None

    def init(self, clients, train_datasets, test_datasets):
        for i in range(len(train_datasets)):
            self.train_labels[i] = np.array(train_datasets[i].targets)
            self.test_labels[i] = np.array(test_datasets[i].targets)
            self.train_data[i] = train_datasets[i].data
            self.test_data[i] = test_datasets[i].data
        self.index_list = self.generate_data(clients, self.train_labels, train_datasets)
        self.test_index_list = self.generate_data(1, self.test_labels, test_datasets, train=False,
                                                  message="test_dataset")

    def generate_data(self, clients_num, labels, datasets, train=True, message="train_dataset"):
        index_list = [[] for _ in range(clients_num)]
        pos = 0
        for i in range(len(datasets)):
            print(f"generating dataset {i}\n")
            sub_list = super().generate_data(clients_num, labels[i], datasets[i], train, message)
            if not train:
                sub_list = [sub_list]
            for c_id, client_list in enumerate(sub_list):
                for j in range(len(client_list)):
                    client_list[j] += pos
                index_list[c_id].append(client_list)
            pos += len(datasets[i])
        if train:
            return index_list
        else:
            return index_list[0]


class MNIST_FM(CompositeAbstractDataset):
    def __init__(self, clients, iid_config, params):
        super().__init__(iid_config)
        transformer = transforms.Compose([
            # 将图片转化为Tensor格式
            transforms.ToTensor()
        ])
        # 获取数据集
        self.names = ["MNIST", "FashionMNIST"]
        train_datasets = [None, None]
        test_datasets = [None, None]
        train_datasets[0] = datasets.MNIST(root=self.path, train=True,
                                           transform=transformer, download=True)
        test_datasets[0] = datasets.MNIST(root=self.path, train=False,
                                          transform=transformer, download=True)
        train_datasets[1] = datasets.FashionMNIST(root=self.path, train=True,
                                                  transform=transformer, download=True)
        test_datasets[1] = datasets.FashionMNIST(root=self.path, train=False,
                                                 transform=transformer, download=True)
        self.init(clients, train_datasets, test_datasets)
        self.train_dataset = CompositeDataset(train_datasets)
        self.test_dataset = CompositeDataset(test_datasets)
