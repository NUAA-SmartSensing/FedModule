import os

import numpy as np

from utils import ModuleFindTool
from utils.IID import generate_data, print_dist


class BaseDataset:
    def __init__(self, iid_config):
        self.data_distribution_generator = None
        self.index_list = None
        self.test_index_list = None
        self.datasets = []
        self.config = iid_config
        self.train_data_size = None
        self.test_data = None
        self.train_data = None
        self.test_dataset = None
        self.train_labels = None
        self.raw_data = None
        self.train_dataset = None
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/')

    def init(self, clients, train_dataset, test_dataset):
        self.raw_data = train_dataset.data
        self.train_labels = np.array(train_dataset.targets)
        self.test_labels = np.array(test_dataset.targets)
        self.train_data = train_dataset.data
        self.test_data = test_dataset.data

        self.train_data_size = self.train_data.shape[0]
        self.index_list = self.generate_data(clients, self.train_labels, train_dataset)
        self.test_index_list = self.generate_data(1, self.test_labels,test_dataset, train=False)

    def get_test_dataset(self):
        return self.test_dataset

    def get_index_list(self):
        return self.index_list

    def get_test_index_list(self):
        return self.test_index_list

    def get_train_dataset(self):
        return self.train_dataset

    def get_config(self):
        return self.config

    def generate_data(self, clients_num, labels, dataset, train=True):
        if train:
            config = self.config
            index_list = self._generate_data(config, labels, clients_num, dataset)
            print(f"train data generation process completed")
        else:
            if isinstance(self.config, dict) and 'test' in self.config:
                test_config = self.config['test']
                index_list = self._generate_data(test_config, labels, 1, dataset)[0]
            else:
                index_list = list(range(len(labels)))
                print_dist([index_list], labels)
            print(f"non-train data generation process completed")
        return index_list

    def _generate_data(self, config, labels, clients_num, dataset):
        if isinstance(config, dict) and "path" in config:
            print("generate customize data distribution...")
            if self.data_distribution_generator is None:
                self.data_distribution_generator = ModuleFindTool.find_class_by_path(self.config["path"])(
                    self.config["params"])
            index_list = self.data_distribution_generator.generate_data(self.config, labels, clients_num, dataset)
        else:
            index_list = generate_data(config, labels, clients_num)
        return index_list
