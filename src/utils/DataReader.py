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


def _get_data(dataset):
    if hasattr(dataset, 'data'):
        return dataset.data
    else:
        raise Exception("Dataset data not found")


def _get_targets(dataset):
    if hasattr(dataset, 'targets'):
        return dataset.targets
    elif hasattr(dataset, 'labels'):
        return dataset.labels
    else:
        raise Exception("Dataset label not found")


class FLDataset(Dataset):
    def __init__(self, dataset, idxs, transform=None, target_transform=None):
        self.dataset = dataset
        self.data = _get_data(dataset)
        self.targets = _get_targets(dataset)
        self.idxs = idxs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.data[self.idxs[item]], self.targets[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

