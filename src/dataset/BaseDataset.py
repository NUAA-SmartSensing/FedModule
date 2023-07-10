import numpy as np

from utils.IID import generate_iid_data, generate_non_iid_data


class BaseDataset:
    def __init__(self, iid_config):
        self.datasets = []
        self.iid_config = iid_config
        self.train_data_size = None
        self.test_data = None
        self.train_data = None
        self.test_datasets = None
        self.train_labels = None
        self.raw_data = None

    def init(self, clients, train_datasets, test_datasets):
        self.raw_data = train_datasets.data
        self.train_labels = np.array(train_datasets.targets)
        self.test_datasets = test_datasets
        self.train_data = train_datasets.data
        self.test_data = test_datasets.data

        self.train_data_size = self.train_data.shape[0]

        if isinstance(self.iid_config, bool):
            print("generating iid data...")
            generate_iid_data(self, clients, train_datasets)
        else:
            print("generating non_iid data...")
            generate_non_iid_data(self.iid_config, self, clients, self.train_labels.min(), self.train_labels.max() + 1,
                                  train_datasets)
        print("data generation process completed")

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.datasets

    def get_config(self):
        return self.iid_config

