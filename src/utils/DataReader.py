import copy
import multiprocessing
import os
import random
import threading
import time

import SharedArray as sa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.GlobalVarGetter import GlobalVarGetter


class LockFactory:
    lock = None

    @staticmethod
    def create_lock():
        if not LockFactory.lock:
            config = GlobalVarGetter().get()['global_config']
            if 'mode' in config and config['mode'] == 'process':
                LockFactory.lock = multiprocessing.Lock()
            else:
                LockFactory.lock = threading.Lock()
        return LockFactory.lock


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


class SharedTensor:
    def __init__(self, protocol_name):
        self.protocol_name = protocol_name
        self.init = False
        self.shared_data = None
        self.shared_np_data = sa.attach(self.protocol_name)
        self.data_process(self.shared_np_data)

    def __len__(self):
        return len(self.shared_data)

    def __getitem__(self, item):
        return self.shared_data[item]

    def data_process(self, shared_data):
        self.shared_data = []
        for i in range(shared_data.shape[0]):
            self.shared_data.append(torch.from_numpy(np.array(shared_data[i])))


class DataReader:
    def __init__(self, dataset):
        self.dataset = dataset
        total_data = [[], []]
        shared_data = [0, 0]
        self.global_var = GlobalVarGetter().get()
        dl = DataLoader(self.dataset, batch_size=1, shuffle=False)
        for batch in dl:
            total_data[0].append(batch[0][0])
            total_data[1].append(batch[1][0])
        data_ndarray = []
        for l in total_data:
            data_ndarray.append(np.array([t.numpy() for t in l]))
        protocol = str(time.time()) + str(os.getpid()) + str(threading.current_thread().ident)

        if 'shared_mem' not in self.global_var:
            self.global_var['shared_mem'] = [f"shm://{protocol}_data", f"shm://{protocol}_target"]
        else:
            self.global_var['shared_mem'].append(f"shm://{protocol}_data")
            self.global_var['shared_mem'].append(f"shm://{protocol}_target")
        shared_data[0] = sa.create(f"shm://{protocol}_data", data_ndarray[0].shape,dtype=data_ndarray[0].dtype)
        shared_data[1] = sa.create(f"shm://{protocol}_target", data_ndarray[1].shape,dtype=data_ndarray[1].dtype)

        for i in range(len(total_data[0])):
            shared_data[0][i] = total_data[0][i]
        for i in range(len(total_data[1])):
            shared_data[1][i] = total_data[1][i]
        self.total_data = []
        self.total_data.append(SharedTensor(f"shm://{protocol}_data"))
        self.total_data.append(SharedTensor(f"shm://{protocol}_target"))


class MultiDataset(Dataset):
    def __init__(self, dataset, index_list, queue_list):
        self.dataset = dataset
        self.index_list = index_list
        self.queue_list = queue_list

    def __getitem__(self, item):
        image, label = self.dataset[self.index_list[item]]
        return image, label
