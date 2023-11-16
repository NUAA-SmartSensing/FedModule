from abc import abstractmethod

from torchvision import transforms


class TransformForCIFARFactory:
    @staticmethod
    def createTransform():
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        )
        return transform

    @staticmethod
    def createTransformWithTensor():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32)])
        return transform
