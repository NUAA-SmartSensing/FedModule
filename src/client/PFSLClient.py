import copy
import time
from client import DLClient
from utils import ModuleFindTool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


def distillation(y, labels, teacher_scores, T, alpha):
    loss1 = F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1), reduction='batchmean') * T * T * 2
    loss2 = F.cross_entropy(y, labels)
    # loss1 = F.mse_loss(y, teacher_scores) * 2
    return loss1 + loss2


class PFSLClient(DLClient.DLClient):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev):
        DLClient.DLClient.__init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config,
                                   dev)
        self.prune_ratio = 1
        # 教师模型
        teacher_model_class = ModuleFindTool.find_class_by_path(config["teacher_model"]["path"])
        for k, v in config["model"]["params"].items():
            if isinstance(v, str):
                config["model"]["params"][k] = eval(v)
        self.teacher_model = teacher_model_class(**config["teacher_model"]["params"])
        self.teacher_model = self.teacher_model.to(self.dev)
        self.alpha = self.config['alpha']

    def run(self):
        while not self.stop_event.is_set():
            self.wait_notify()

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                if self.time_stamp == -1:
                    # 预训练
                    data_sum, weights = self.pre_train()
                else:
                    # 该client进行训练
                    data_sum, weights = self.train()
                    # client传回server的信息具有延迟
                    # 本地测试
                    self.run_test()
                    time.sleep(0.01)

                # 返回其ID、模型参数和时间戳
                self.upload(data_sum, weights)
                self.event.clear()

                self.message_queue.set_training_status(self.client_id, False)
            # 该client等待被选中
            else:
                self.event.wait()
                self.message_queue.set_training_status(self.client_id, True)

    def pre_train(self):
        # 设置迭代次数
        data_sum = 0
        for epoch in range(2):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = self.model(data)
                # 计算损失函数
                loss = self.loss_func(preds, label)
                data_sum += label.size(0)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                self.opti.step()
                # 将梯度归零，初始化梯度
                self.opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(self.model.state_dict())
        # 根据delay计算模型剪枝率
        self.prune_ratio = 0.3 + self.delay / 6 * 0.4
        self.model.pruning_by_ratio(self.prune_ratio)
        print("client id:", self.client_id, "pruning over,remain", round((1 - self.prune_ratio), 2) * 100, "%")
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights

    def train(self):
        # 设置迭代次数
        data_sum = 0
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                output = self.model(data)
                # 教师模型推理
                teacher_output = self.teacher_model(data)
                teacher_output = teacher_output.detach()
                # 计算损失函数
                loss = distillation(output, label, teacher_output, T=20, alpha=self.config['alpha'])
                data_sum += label.size(0)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                self.opti.step()
                # 将梯度归零，初始化梯度
                self.opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        weights = copy.deepcopy(self.model.state_dict())
        for k, v in weights.items():
            weights[k] = weights[k].cpu().detach()
        return data_sum, weights

    def run_test(self):
        super().run_test()
        return self.loss_list[len(self.loss_list) - 1], self.accuracy_list[len(self.accuracy_list) - 1]

    def upload(self, data_sum, weights):
        if self.time_stamp != -1:
            update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                           "time_stamp": self.time_stamp, "prune_ratio": self.prune_ratio,
                           "accuracy": self.accuracy_list[len(self.accuracy_list) - 1],
                           "loss": self.loss_list[len(self.loss_list) - 1]}
            self.message_queue.put_into_uplink(update_dict)

    def wait_notify(self):
        if self.message_queue.get_from_downlink(self.client_id, 'received_time_stamp'):
            self.message_queue.put_into_downlink(self.client_id, 'received_time_stamp', False)
            self.time_stamp = self.message_queue.get_from_downlink(self.client_id, 'time_stamp_buffer')
            self.schedule_t = self.message_queue.get_from_downlink(self.client_id, 'schedule_time_stamp_buffer')

        if self.message_queue.get_from_downlink(self.client_id, 'received_weights'):
            self.message_queue.put_into_downlink(self.client_id, 'received_weights', False)
            self.weights_buffer = self.message_queue.get_from_downlink(self.client_id, 'weights_buffer')
            # 更新模型参数
            if self.time_stamp == -1:
                self.model.load_state_dict(self.weights_buffer['global'], strict=True)
                self.received_weights = False
            else:
                # 获取教师模型
                if self.init:
                    if self.client_id in self.weights_buffer.keys():
                        # self.model.load_state_dict(self.weights_buffer[self.client_id], strict=True)
                        self.teacher_model.load_state_dict(self.weights_buffer[self.client_id], strict=True)
                else:
                    self.teacher_model.load_state_dict(copy.deepcopy(self.model.state_dict()), strict=True)
                    self.init = True
                self.received_weights = False

    def run_model_test(self, model):
        test_correct = 0
        for data in self.test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(self.dev), labels.to(self.dev)
            outputs = model(inputs)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == labels.data).cpu().numpy()
        accuracy = (test_correct * 100) / (len(self.test_dl) * self.config['test_batch_size'])
        return accuracy
