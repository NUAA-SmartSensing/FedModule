import copy

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from utils.Tools import *
from utils.JsonTool import *


class FashionMNIST:
    def __init__(self, clients, iid_config):
        # 获取数据集
        train_datasets = datasets.FashionMNIST(root='../data/', train=True,
                                               transform=transforms.ToTensor(), download=True)
        test_datasets = datasets.FashionMNIST(root='../data/', train=False,
                                              transform=transforms.ToTensor(), download=True)
        train_data = train_datasets.data
        self.raw_data = train_datasets.data
        self.train_labels = train_datasets.targets
        test_data = test_datasets.data
        self.test_datasets = test_datasets
        # 归一化
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1] * train_data.shape[2])
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1] * test_data.shape[2])
        train_data = train_data.float()
        self.train_data = np.multiply(train_data, 1.0 / 255.0)  # 数组对应元素位置相乘
        test_data = test_data.float()
        self.test_data = np.multiply(test_data, 1.0 / 255.0)

        self.train_data_size = train_data.shape[0]
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
                self.datasets.append(TensorDataset(torch.tensor(client_data), torch.tensor(client_label)))
        else:
            print("generating non_iid data...")
            label_config = iid_config['label']
            data_config = iid_config['data']
            # 生成label lists
            if isinstance(label_config, dict):
                # step
                if "step" in label_config.keys():
                    label_lists = generate_label_lists_by_step(label_config["step"], label_config["list"], 0, 10)
                # {[],[],[]}
                else:
                    label_lists = dict_to_list(label_config)
            # []
            else:
                label_lists = generate_label_lists(label_config, 0, 10)

            # 生成data lists
            # {}
            if len(data_config) == 0:
                size = self.train_data_size // clients
                data_lists = generate_data_lists(size, size, clients, label_lists)
            # max,min
            else:
                data_lists = generate_data_lists(data_config["max"], data_config["min"], clients, label_lists)
            # 生成datasets
            self.datasets = generate_non_iid_data(self.train_data, self.train_labels, label_lists, data_lists)
            # 保存label至配置文件
            self.iid_config['label'] = list_to_dict(label_lists)
        print("data generation process completed")

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.datasets

    def get_config(self):
        return self.iid_config
