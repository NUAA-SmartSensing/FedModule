import time

import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from client import NormalClient
from utils.DataReader import FLDataset
from utils.Tools import saveAns


class TestClient(NormalClient.NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev):
        NormalClient.NormalClient.__init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev)
        self.train_ds, self.test_dataset = train_test_split(FLDataset(train_ds, index_list), test_size=config['test_size'])
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        # 提供给wandb使用
        self.step = 1
        # 本地数据存储
        self.accuracy_list = []
        self.loss_list = []

    def run(self):
        self.global_config = self.message_queue.get_config("global_config")
        while not self.stop_event.is_set():
            if self.message_queue.get_from_downlink(self.client_id, 'received_weights'):
                self.message_queue.put_into_downlink(self.client_id, 'received_weights', False)
                # 更新模型参数
                self.model.load_state_dict(self.message_queue.get_from_downlink(self.client_id, 'weights_buffer'), strict=True)
            if self.message_queue.get_from_downlink(self.client_id, 'received_time_stamp'):
                self.message_queue.put_into_downlink(self.client_id, 'received_time_stamp', False)
                self.time_stamp = self.message_queue.get_from_downlink(self.client_id, 'time_stamp_buffer')
                self.schedule_t = self.message_queue.get_from_downlink(self.client_id, 'schedule_time_stamp_buffer')

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                # 该client进行训练

                data_sum, weights = self.train()
                # client传回server的信息具有延迟
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
        dl = DataLoader(self.test_dataset, batch_size=self.config['test_batch_size'], shuffle=True, drop_last=True)
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
        print("Client", self.client_id, "trained, accuracy:", accuracy, 'loss', loss)
        if 'wandb' in self.config and self.config['wandb']:
            wandb.log({f'{self.client_id}_accuracy': accuracy, f'{self.client_id}_loss': loss, f'time_stamp': self.time_stamp, f'local_epoch': self.step})
            self.step += 1
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)
