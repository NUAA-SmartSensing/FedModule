import time

import numpy
import torch
import wandb
from torch.utils.data import DataLoader

from client import NormalClient
from utils import ModuleFindTool
from utils.ModelTraining import train_one_epoch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DLClient(NormalClient.NormalClient):
    def __init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev,  print_lock, global_var):
        NormalClient.NormalClient.__init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev, print_lock,  global_var)
        self.init = False
        self.train_dataset, self.test_dataset = train_test_split(train_ds, test_size=client_config['test_size'])
        self.train_dl = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def run(self):
        while not self.stop_event.is_set():
            if self.received_weights:
                # 更新模型参数
                if self.init:
                    if self.client_id in self.weights_buffer.keys():
                        for key, var in self.model.state_dict().items():
                            if torch.cuda.is_available():
                                self.weights_buffer[self.client_id][key] = self.weights_buffer[self.client_id][key].to(self.dev)
                            self.weights_buffer[self.client_id][key] = self.config['alpha'] * var + (1 - self.config['alpha']) * self.weights_buffer[self.client_id][key]
                        self.model.load_state_dict(self.weights_buffer[self.client_id], strict=True)
                else:
                    self.model.load_state_dict(self.weights_buffer['global'], strict=True)
                    self.init = True
                self.received_weights = False
            if self.received_time_stamp:
                self.time_stamp = self.time_stamp_buffer
                self.received_time_stamp = False
            if self.event_is_set:
                self.event_is_set = False

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.client_thread_lock.acquire()
                # 该client进行训练
                data_sum, weights = train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func,
                                                    self.opti, self.mu)
                self.print_lock.acquire()
                print("Client", self.client_id, data_sum)
                self.print_lock.release()
                # client传回server的信息具有延迟
                # 本地测试
                self.run_server_test()
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                               "time_stamp": self.time_stamp}
                self.queue_manager.put(update_dict)
                self.event.clear()
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()

    def run_server_test(self):
        dl = DataLoader(self.test_dataset, batch_size=self.config['test_batch_size'], shuffle=True)
        test_correct = 0
        test_loss = 0
        for data in dl:
            inputs, labels = data
            inputs, labels = inputs.to(self.dev), labels.to(self.dev)
            outputs = self.model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_loss += self.loss_func(outputs, labels).item()
        accuracy = (test_correct * 100) / (len(dl) * self.config['test_batch_size'])
        loss = test_loss / len(dl)
        self.print_lock.acquire()
        print("Client", self.client_id, "trained, accuracy:", accuracy, 'loss', loss)
        self.print_lock.release()
        if self.global_var['updater'].config['enabled']:
            wandb.log({f'{self.client_id}_accuracy': accuracy, f'{self.client_id}_loss': loss})
