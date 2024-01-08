import time

import torch
import wandb
from torch.utils.data import DataLoader

from client.NormalClient import NormalClient
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.DataReader import FLDataset
from utils.Tools import saveAns


class TestClient(NormalClient):
    def __init__(self, c_id, init_lock, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, init_lock, stop_event, selected_event, delay, index_list, config, dev)
        self.fl_test_ds = None
        self.test_dl = None
        self.global_config = None
        test_size = config['test_size']
        n1 = int(len(index_list) * test_size)
        n2 = len(index_list) - n1
        self.test_index_list, self.train_index_list = torch.utils.data.random_split(index_list, [n1, n2])

        # 提供给wandb使用
        self.step = 1
        # 本地数据存储
        self.accuracy_list = []
        self.loss_list = []

    def run(self):
        self.global_config = self.message_queue.get_config("global_config")
        self.init_client()
        while not self.stop_event.is_set():
            self.wait_notify()

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                # 该client进行训练
                data_sum, weights = self.train()

                # client传回server的信息具有延迟
                # 本地测试
                self.run_test()
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                self.upload(data_sum, weights)
                self.event.clear()

                self.message_queue.set_training_status(self.client_id, False)
            # 该client等待被选中
            else:
                self.event.wait()
                self.message_queue.set_training_status(self.client_id, True)
        saveAns(f'../results/{self.global_config["experiment"]}/{self.client_id}_accuracy.txt', list(self.accuracy_list))
        saveAns(f'../results/{self.global_config["experiment"]}/{self.client_id}_loss.txt', list(self.loss_list))

    def run_test(self):
        test_correct = 0
        test_loss = 0
        for data in self.test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(self.dev), labels.to(self.dev)
            outputs = self.model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_loss += self.loss_func(outputs, labels).item()
        accuracy = (test_correct * 100) / (len(self.test_dl) * self.config['test_batch_size'])
        loss = test_loss / len(self.test_dl)
        print("Client", self.client_id, "trained, accuracy:", accuracy, 'loss', loss)
        if 'wandb' in self.config and self.config['wandb']:
            wandb.log({f'{self.client_id}_accuracy': accuracy, f'{self.client_id}_loss': loss, f'time_stamp': self.time_stamp, f'local_epoch': self.step})
            self.step += 1
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)

    def init_client(self):
        config = self.config
        # transform
        if "transform" in config:
            transform_func = ModuleFindTool.find_class_by_path(config["transform"]["path"])
            self.transform = transform_func(**config["transform"]["params"])
        if "target_transform" in config:
            target_transform_func = ModuleFindTool.find_class_by_path(config["target_transform"]["path"])
            self.target_transform = target_transform_func(**config["target_transform"]["params"])

        # 本地模型
        model_class = ModuleFindTool.find_class_by_path(config["model"]["path"])
        for k, v in config["model"]["params"].items():
            if isinstance(v, str):
                config["model"]["params"][k] = eval(v)
        self.model = model_class(**config["model"]["params"])
        self.model = self.model.to(self.dev)

        self.train_ds = self.message_queue.get_train_dataset()
        self.fl_train_ds = FLDataset(self.train_ds, list(self.index_list), self.transform, self.target_transform)
        self.fl_test_ds = FLDataset(self.train_ds, list(self.test_index_list), self.transform, self.target_transform)

        # 优化器
        opti_class = ModuleFindTool.find_class_by_path(self.optimizer_config["path"])
        self.opti = opti_class(self.model.parameters(), **self.optimizer_config["params"])

        # loss函数
        self.loss_func = LossFactory(config["loss"], self).create_loss()

        self.train_dl = DataLoader(self.fl_train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_dl = DataLoader(self.fl_test_ds, batch_size=self.config['test_batch_size'], shuffle=True, drop_last=True)
