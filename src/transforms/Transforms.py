from abc import abstractmethod

from torchvision import transforms


class TransformForCIFARFactory:
    @abstractmethod
    def createTransform(self):
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32)])
        return transform
