import copy
import threading
from abc import abstractmethod

import torch.utils.data
import wandb
from torch.utils.data import DataLoader

from loss.LossFactory import LossFactory
from update.UpdateCaller import UpdateCaller
from utils import ModuleFindTool
from utils.DataReader import CustomDataset
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import MessageQueueFactory


def _read_data(dataset):
    data = []
    targets = []
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for x, y in dl:
        data.append(x[0])
        targets.append(y[0])
    return data, targets


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
        self.test_data = self._get_test_dataset(test_data)

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

        self.message_queue = MessageQueueFactory.create_message_queue()
        self.optimizer = None
        # server_opt
        if "optimizer" in self.config:
            self.optimizer = ModuleFindTool.find_class_by_path(self.config['optimizer']['path'])(
                self.server_network.parameters(), **self.config["optimizer"]["params"])

    @abstractmethod
    def run(self):
        pass

    def update_server_weights(self, epoch, update_list):
        global_model, delivery_weights = self.update_caller.update_server_weights(epoch, update_list)
        if torch.cuda.is_available():
            for key, var in global_model.items():
                global_model[key] = global_model[key].cuda()
        new_global_model = self.update_global_model(global_model)
        # process the PFL
        if delivery_weights != global_model:
            self.set_delivery_weights(delivery_weights)
        else:
            self.set_delivery_weights(new_global_model)

    def run_server_test(self, epoch):
        dl = DataLoader(self.test_data, batch_size=100, shuffle=True, drop_last=True)
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

    def set_delivery_weights(self, weights):
        self.global_var['scheduler'].server_weights = copy.deepcopy(weights)

    def update_global_model(self, new_model):
        if self.optimizer is not None:
            training_params = self.message_queue.get_training_params()
            global_model = self.server_network.state_dict()
            g = {}
            for k in global_model:
                if training_params[k]:
                    g[k] = new_model[k] - global_model[k]
            for k, w in zip(g, self.server_network.parameters()):
                w.grad = -g[k]
            self.optimizer.step()
        else:
            self.server_network.load_state_dict(new_model)
        return self.server_network.state_dict()

    def _get_test_dataset(self, test_data):
        # 预加载
        if 'dataset_pre_load' in self.global_var['global_config'] and self.global_var['global_config']['dataset_pre_load']:
            data, targets = _read_data(test_data)
            return CustomDataset(data, targets)
        # 静态加载
        else:
            return test_data
