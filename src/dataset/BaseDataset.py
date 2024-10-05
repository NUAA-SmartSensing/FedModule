import os

import numpy as np

from utils import ModuleFindTool
from utils.IID import generate_iid_data, generate_non_iid_data


class BaseDataset:
    def __init__(self, iid_config):
        self.data_distribution_generator = None
        self.index_list = None
        self.test_index_list = None
        self.label_min = None
        self.label_max = None
        self.datasets = []
        self.iid_config = iid_config
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
        self.label_max = self.train_labels.max()
        self.label_min = self.train_labels.min()

        self.train_data_size = self.train_data.shape[0]
        self.index_list = self.generate_data(clients, self.train_labels,train_dataset)
        self.test_index_list = self.generate_data(1, self.test_labels,test_dataset, train=False, message="test_dataset")

    def get_test_dataset(self):
        return self.test_dataset

    def get_index_list(self):
        return self.index_list

    def get_test_index_list(self):
        return self.test_index_list

    def get_train_dataset(self):
        return self.train_dataset

    def get_config(self):
        return self.iid_config

    def generate_data(self, clients_num, labels, dataset, train=True, message="train_dataset"):
        if isinstance(self.iid_config, bool):
            print("generating iid data...")
            index_list = generate_iid_data(labels, clients_num)
        elif isinstance(self.iid_config, dict) and "path" in self.iid_config:
            print("generate customize data distribution...")
            if self.data_distribution_generator is None:
                self.data_distribution_generator = ModuleFindTool.find_class_by_path(self.iid_config["path"])(self.iid_config["params"])
            index_list = self.data_distribution_generator.generate_data(self.iid_config, labels, clients_num, dataset, train)
        else:
            print("generating non_iid data...")
            index_list = generate_non_iid_data(self.iid_config, labels, clients_num, train)
        print(f"{message} data generation process completed")
        if train:
            return index_list
        else:
            return index_list[0]
