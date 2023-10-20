from abc import abstractmethod

from torchvision import transforms


class TransformForCIFARFactory:
    @staticmethod
    def createTransform():
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32)])
        return transform

    @staticmethod
    def createTransformWithTensor():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32)])
        return transform
