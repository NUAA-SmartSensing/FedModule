from torch.utils.data import DataLoader, Dataset

from utils.GlobalVarGetter import GlobalVarGetter


class FLDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[0][self.idxs[item]], self.dataset[1][self.idxs[item]]
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
