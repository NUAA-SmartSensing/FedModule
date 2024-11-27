import random

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from core.Component import Component
from core.handlers.Handler import Handler
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import saveAns


class ModelEvaluateHandler(Handler):
    def __init__(self):
        super().__init__()
        global_var = GlobalVarGetter.get()
        cloud_enabled = global_var['config']['wandb']['enabled']
        file_enabled = global_var['config']['global']['save']
        path = global_var['config']['global']['test_method'] if 'test_method' in global_var['config']['global'] else 'core.handlers.ModelEvaluateHandler._DefaultTest'
        self.test_method = ModuleFindTool.find_class_by_path(path)(file_enabled, cloud_enabled)
        self.loss_list = []
        self.accuracy_list = []

    def __split_test_data(self, lst, length):
        random.shuffle(lst)
        lst = np.array(lst)
        return np.split(lst, length)

    def _handle(self, request):
        if 'updater' in request:
            updater = request.get('updater')
            test_data = updater.test_data
            dev = updater.dev
            model = updater.model
            loss_func = updater.loss_func
        else:
            client = request.get('client')
            dev = client.dev
            model = client.model
            loss_func = client.loss_func
            test_data = client.test_data
        epoch = request.get('epoch')
        self.test_method.test(test_data, model, loss_func, dev, epoch)
        return request

    def run_once(self, request):
        experiment = request.get('global_var')['config']['global']['experiment']
        if 'updater' in request:
            updater: Component = request.get('updater')
            updater.add_final_callback(self.test_method.callback, f'../results/{experiment}/accuracy.txt')
        else:
            client = request.get('client')
            client_id = client.client_id
            client.add_final_callback(self.test_method.callback, f'../results/{experiment}/{client_id}_accuracy.txt')
            if not hasattr(client, 'test_data'):
                test_size = client.config['test_size'] if 'test_size' in client.config else 0.1
                split_data = self.__split_test_data(client.train_data, test_size)
                test_data = FLDataset(client.train_ds, list(split_data), client.transform, client.target_transform)
                client.test_data = test_data


class _AbstractTest:
    def __init__(self, file_enabled, cloud_enabled):
        self.file_enabled = file_enabled
        self.cloud_enabled = cloud_enabled

    def test(self, test_data, model, loss_func, dev, epoch):
        pass

    def callback(self, path):
        pass


class _DefaultTest(_AbstractTest):
    def __init__(self, file_enabled, cloud_enabled):
        super().__init__(file_enabled, cloud_enabled)
        self.accuracy_list = []
        self.loss_list = []

    def test(self, test_data, model, loss_func, dev, epoch):
        dl = DataLoader(test_data, batch_size=100, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        with torch.no_grad():
            for data in dl:
                inputs, labels = data
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = model(inputs)
                _, id = torch.max(outputs.data, 1)
                test_loss += loss_func(outputs, labels).detach().item()
                test_correct += torch.sum(id == labels.data).cpu().numpy()
            accuracy = test_correct / len(dl)
            loss = test_loss / len(dl)
            self.loss_list.append(loss)
            self.accuracy_list.append(accuracy)
            if self.cloud_enabled:
                wandb.log({'accuracy': accuracy, 'loss': loss}, step=epoch)
            print('Epoch(t):', epoch, 'accuracy:', accuracy, 'loss', loss)

    def callback(self, path):
        if self.file_enabled:
            saveAns(path, list(self.accuracy_list))
            saveAns(path, list(self.loss_list))
