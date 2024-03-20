from abc import abstractmethod

from torchvision import transforms


class TransformForCIFARFactory:
    @staticmethod
    def createTransform():
        return transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        )

    @staticmethod
    def createTransformWithTensor():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32)])


class ToTensorFactory:
    @staticmethod
    def createToTensor():
        return transforms.ToTensor()
