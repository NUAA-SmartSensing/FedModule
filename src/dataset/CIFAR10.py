from torchvision import datasets, transforms

from dataset.BaseDataset import BaseDataset


class CIFAR10(BaseDataset):
    def __init__(self, clients, config, params):
        BaseDataset.__init__(self, config)
        mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
        # These values are mostly used by researchers as found to very useful in fast convergence
        # img_size=224
        # crop_size = 224
        # transformTrain = transforms.Compose(
        #     [
        #     # transforms.Resize(img_size),#, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        #     #transforms.CenterCrop(crop_size),
        #     # transforms.RandomRotation(20),
        #     # transforms.RandomHorizontalFlip(0.1),
        #     # transforms.ColorJitter(brightness=0.1,contrast = 0.1 ,saturation =0.1 ),
        #     # transforms.RandomAdjustSharpness(sharpness_factor = 2, p = 0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean,std),
        #     # transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
        #     ]
        # )

        # transformTest = transforms.Compose(
        # [
        #     # transforms.Resize((img_size,img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)])
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        # 获取数据集
        self.train_dataset = datasets.CIFAR10(root=self.path, train=True,
                                              transform=transformer, download=True)
        self.test_dataset = datasets.CIFAR10(root=self.path, train=False,
                                             transform=transformer, download=True)
        self.init(clients, self.train_dataset, self.test_dataset)
