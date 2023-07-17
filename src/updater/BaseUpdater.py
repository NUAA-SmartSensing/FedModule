import threading
from abc import abstractmethod

import numpy as np
import torch.utils.data
import wandb
from torch.utils.data import DataLoader

from loss.LossFactory import LossFactory
from update.UpdateCaller import UpdateCaller
from utils import ModuleFindTool
from utils.DataReader import DataReader, FLDataset
from utils.GlobalVarGetter import GlobalVarGetter


class BaseUpdater(threading.Thread):
    def __init__(self, server_thread_lock, stop_event, config):
        threading.Thread.__init__(self)
        self.server_thread_lock = server_thread_lock
        self.stop_event = stop_event
        self.config = config
        self.global_var = GlobalVarGetter().get()

        self.T = self.global_var['T']
        self.current_time = self.global_var['current_t']
        self.schedule_t = self.global_var['schedule_t']
        self.server_network = self.global_var['server_network']
        self.client_manager = self.global_var['client_manager']

        test_data = self.global_var['dataset'].get_test_dataset()
        data_reader = DataReader(test_data)
        self.test_data = FLDataset(data_reader.total_data, np.arange(len(data_reader.total_data[0])))

        self.queue_manager = self.global_var['queue_manager']
        self.print_lock = self.global_var['print_lock']

        self.event = threading.Event()
        self.event.clear()

        self.sum_delay = 0

        self.accuracy_list = []
        self.loss_list = []

        # loss函数
        self.loss_func = LossFactory(self.config['loss']).create_loss()

        # 聚合算法
        update_class = ModuleFindTool.find_class_by_path(self.config['update']['path'])
        self.update_method = update_class(self.config['update']['params'])
        self.update_caller = UpdateCaller(self)

    @abstractmethod
    def run(self):
        pass

    def update_server_weights(self, epoch, update_list):
        updated_parameters = self.update_caller.update_server_weights(epoch, update_list)
        for key, var in updated_parameters.items():
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        self.server_network.load_state_dict(updated_parameters)

    def run_server_test(self, epoch):
        dl = DataLoader(self.test_data, batch_size=64, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        for data in dl:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = self.server_network(inputs)
            _, id = torch.max(outputs.data, 1)
            test_loss += self.loss_func(outputs, labels).item()
            test_correct += torch.sum(id == labels.data).cpu().numpy()
        accuracy = test_correct / len(dl)
        loss = test_loss / len(dl)
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)
        self.print_lock.acquire()
        print('Epoch(t):', epoch, 'accuracy:', accuracy, 'loss', loss)
        if self.config['enabled']:
            wandb.log({'accuracy': accuracy, 'loss': loss})
        self.print_lock.release()
        return accuracy, loss

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list
