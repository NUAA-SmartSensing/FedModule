from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset


class CIFAR100(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 获取数据集
        self.train_dataset = datasets.CIFAR100(root=self.path, train=True,
                                               transform=transformer, download=True)
        self.test_dataset = datasets.CIFAR100(root=self.path, train=False,
                                              transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)
