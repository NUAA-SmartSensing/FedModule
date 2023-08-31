from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset


class FashionMNIST(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            # 将图片转化为Tensor格式
            transforms.ToTensor(),
            # 正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 获取数据集
        self.train_dataset = datasets.FashionMNIST(root=self.path, train=True,
                                                   transform=transformer, download=True)
        self.test_dataset = datasets.FashionMNIST(root=self.path, train=False,
                                                  transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)
