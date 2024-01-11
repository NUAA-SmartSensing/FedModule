from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset


class EMNIST(BaseDataset):
    def __init__(self, clients, iid_config, params):
        BaseDataset.__init__(self, iid_config)
        transformer = transforms.Compose([
            # 将图片转化为Tensor格式
            transforms.ToTensor(),
        ])
        self.params = params
        # 获取数据集
        self.train_dataset = datasets.EMNIST(root=self.path, train=True,
                                             transform=transformer, download=True, split=self.params["split"])
        self.test_dataset = datasets.EMNIST(root=self.path, train=False,
                                            transform=transformer, download=True, split=self.params["split"])
        self.init(clients, self.train_dataset, self.test_dataset)
