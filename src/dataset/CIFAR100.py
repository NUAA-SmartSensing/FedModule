from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset


class CIFAR100(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]),
        ])
        # 获取数据集
        self.train_dataset = datasets.CIFAR100(root=self.path, train=True,
                                               transform=transformer, download=True)
        self.test_dataset = datasets.CIFAR100(root=self.path, train=False,
                                              transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)
