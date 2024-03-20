from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


class FLDataset(Dataset):
    def __init__(self, dataset, idxs, transform=None, target_transform=None):
        self.dataset = dataset
        self.idxs = idxs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

