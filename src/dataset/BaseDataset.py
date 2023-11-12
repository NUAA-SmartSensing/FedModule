import os

import numpy as np

from utils.IID import generate_iid_data, generate_non_iid_data


class BaseDataset:
    def __init__(self, iid_config):
        self.index_list = None
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
        self.train_data = train_dataset.data
        self.test_data = test_dataset.data
        self.label_max = self.train_labels.max()
        self.label_min = self.train_labels.min()

        self.train_data_size = self.train_data.shape[0]

        if isinstance(self.iid_config, bool):
            print("generating iid data...")
            self.index_list = generate_iid_data(self, clients)
        else:
            print("generating non_iid data...")
            self.index_list = generate_non_iid_data(self.iid_config, self, clients, self.label_min, self.label_max + 1,
                                                    train_dataset)
        print("data generation process completed")

    def get_test_dataset(self):
        return self.test_dataset

    def get_index_list(self):
        return self.index_list

    def get_train_dataset(self):
        return self.train_dataset

    def get_config(self):
        return self.iid_config

