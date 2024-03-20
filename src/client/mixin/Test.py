import random

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from utils.DataReader import FLDataset
from utils.Tools import saveAns


class TestableMixin:
    r"""
    this mixin is used to generate test dataloader and run test. Inherit from this mixin, you can use the following
    methods:

    - generate_test_dl: generate test dataloader and return new train index list
    - run_test: run test and save the record if wandb is True
    - save_test_record: save the test record

    [NOTE]
    please replace the index_list with the new train index list after calling the generate_test_dl method.
    If not, the index_list will be the same as the original index_list, which will cause the test data to be used
    in local training.
    """
    def __init__(self, config):
        self.wandb = 'wandb' in config and config['wandb']
        self.test_dl = None
        self.wandb = False
        self.test_every = config['test_every'] if 'test_every' in config else 1
        self.test_batch_size = config['test_batch_size'] if 'test_batch_size' in config else 64
        test_size = config['test_size'] if 'test_size' in config else 0.1
        self.cnt = -1
        # 提供给wandb使用
        self.step = 1
        # 本地数据存储
        self.accuracy_list = []
        self.loss_list = []
        if not hasattr(self, "index_list"):
            raise AttributeError("Please ensure self.index_list exists")
        n1 = int(len(self.index_list) * test_size)
        self.test_index_list, self.index_list = self.__split_list(self.index_list, [n1])

    @staticmethod
    def __split_list(lst, length):
        random.shuffle(lst)
        lst = np.array(lst)
        return np.split(lst, length)

    def run_test(self, client_id: int, time_stamp: int, model: torch.nn.Module, loss_func: torch.nn.Module, dev):
        """
        run test and save the record if wandb is True

        Parameters
        ----------
        client_id: int
            client id
        time_stamp: int
            time stamp
        model: torch.nn.Module
            model
        loss_func: torch.nn.Module
            loss function
        dev: torch.device or str
            device
        """
        self.cnt += 1
        if self.cnt % self.test_every != 0:
            return
        if self.test_dl is None:
            fl_test_ds = FLDataset(self.train_ds, list(self.test_index_list), self.transform, self.target_transform)
            self.test_dl = DataLoader(fl_test_ds, batch_size=self.test_batch_size, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        for data in self.test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_loss += loss_func(outputs, labels).item()
        accuracy = (test_correct * 100) / (len(self.test_dl) * self.test_batch_size)
        loss = test_loss / len(self.test_dl)
        print("Client", client_id, "trained, accuracy:", accuracy, 'loss', loss)
        if self.wandb:
            wandb.log(
                {f'{client_id}_accuracy': accuracy, f'{client_id}_loss': loss, f'time_stamp': time_stamp,
                 f'local_epoch': self.step})
            self.step += 1
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)

    def save_test_record(self, client_id: int, experiment: str):
        """
        Save the test record

        Parameters
        ----------
        client_id: int
            client id
        experiment: str
            the name of the experiment: global_config['experiment']
        """
        saveAns(f'../results/{experiment}/{client_id}_accuracy.txt', list(self.accuracy_list))
        saveAns(f'../results/{experiment}/{client_id}_loss.txt', list(self.loss_list))

