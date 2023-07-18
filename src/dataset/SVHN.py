import numpy as np
from torchvision import transforms, datasets

from dataset.BaseDataset import BaseDataset
from utils.IID import generate_iid_data, generate_non_iid_data


class SVHN(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            # 将图片转化为Tensor格式
            transforms.ToTensor()
        ])
        # 获取数据集
        self.train_dataset = datasets.SVHN(root='../data/', transform=transformer, download=True)
        self.test_dataset = datasets.SVHN(root='../data/', transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)

    def init(self, clients, train_dataset, test_dataset):
        self.raw_data = train_dataset.data
        self.train_labels = np.array(train_dataset.labels)
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
