import threading
from abc import abstractmethod

import torch.utils.data
import wandb
from torch.utils.data import DataLoader

from core.MessageQueue import MessageQueueFactory
from loss.LossFactory import LossFactory
from update.UpdateCaller import UpdateCaller
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
from utils.Tools import to_cpu, to_dev, random_seed_set


class BaseUpdater(threading.Thread):
    def __init__(self, server_thread_lock, stop_event, config):
        threading.Thread.__init__(self)
        self.server_thread_lock = server_thread_lock
        self.stop_event = stop_event
        self.config = config
        self.global_var = GlobalVarGetter.get()
        random_seed_set(self.global_var['global_config']['seed'])

        self.T = self.global_var['T']
        self.current_time = self.global_var['current_t']
        self.schedule_t = self.global_var['schedule_t']
        self.server_network = self.global_var['server_network']

        self.message_queue = MessageQueueFactory.create_message_queue()
        self.test_data = self.message_queue.get_test_dataset()

        self.queue_manager = self.global_var['queue_manager']
        self.sum_delay = 0

        self.accuracy_list = []
        self.loss_list = []

        # loss function
        self.loss_func = LossFactory(self.config['loss']).create_loss()

        # aggregation method
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
        new_global_model = self.update_global_model(global_model)
        # process the PFL
        if delivery_weights is not None:
            self.set_delivery_weights(delivery_weights)
        else:
            self.set_delivery_weights(new_global_model)

    def run_server_test(self, epoch):
        dl = DataLoader(self.test_data, batch_size=100, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for data in dl:
                inputs, labels = data
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = self.server_network(inputs)
                _, id = torch.max(outputs.data, 1)
                test_loss += self.loss_func(outputs, labels).detach().item()
                test_correct += torch.sum(id == labels.data).cpu().numpy()
            accuracy = test_correct / len(dl)
            loss = test_loss / len(dl)
            self.loss_list.append(loss)
            self.accuracy_list.append(accuracy)
            print('Epoch(t):', epoch, 'accuracy:', accuracy, 'loss', loss)
            if self.config['enabled']:
                wandb.log({'accuracy': accuracy, 'loss': loss})
        return accuracy, loss

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def set_delivery_weights(self, weights):
        self.global_var['scheduler'].server_weights = weights

    def update_global_model(self, new_model):
        new_model = to_dev(new_model, 'cuda')
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
        weights = self.server_network.state_dict()
        return to_cpu(weights)
