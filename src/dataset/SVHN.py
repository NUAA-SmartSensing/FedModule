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
        self.train_dataset = datasets.SVHN(root=self.path, transform=transformer, download=True)
        self.test_dataset = datasets.SVHN(root=self.path, transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)
