from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset


class CIFAR10(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 获取数据集
        train_datasets = datasets.CIFAR10(root='../data/', train=True,
                                          transform=transformer, download=True)
        test_datasets = datasets.CIFAR10(root='../data/', train=False,
                                         transform=transformer, download=True)
        self.init(clients, train_datasets, test_datasets)
