from torch.utils.data import DataLoader, Dataset

from utils.GlobalVarGetter import GlobalVarGetter


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
        self.data = dataset[0]
        self.targets = dataset[1]
        self.idxs = idxs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[0][self.idxs[item]], self.dataset[1][self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class DataReader:
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.total_data = [[], []]
        self.global_var = GlobalVarGetter().get()
        dl = DataLoader(self.dataset, batch_size=1, shuffle=False)
        for batch in dl:
            self.total_data[0].append(batch[0][0])
            self.total_data[1].append(batch[1][0])


class MultiDataset(Dataset):
    def __init__(self, dataset, index_list, queue_list):
        self.dataset = dataset
        self.index_list = index_list
        self.queue_list = queue_list

    def __getitem__(self, item):
        image, label = self.dataset[self.index_list[item]]
        return image, label
