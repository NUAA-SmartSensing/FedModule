from copy import deepcopy

import torch
from torch.nn import functional as F

from client.NormalClient import NormalClient
from utils import ModuleFindTool
from utils.IID import generate_iid_data, generate_non_iid_data


class SelfStreamClient(NormalClient):
    """
    StreamClient is a class that inherits from NormalClient and is used to implement the stream data scenario.
    This class's data is divided into multiple tasks equally, and the model is trained on each task in turn.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.task_num = config["task_num"] if "task_num" in config else 1
        self.task_interval = config["task_interval"] if "task_interval" in config else 1
        self.task_id = 0
        self.total_epoch = 0

    def change_task(self):
        self.fl_train_ds.change_idxs(list(self.index_list)[self.task_id::self.task_num])
        self.task_id = (self.task_id + 1) % self.task_num

    def local_task(self):
        self._check_and_change_task()
        super().local_task()
        self._increment_epoch_count()

    def _check_and_change_task(self):
        if self.total_epoch % self.task_interval == 0:
            print(f"Client {self.client_id} changed task to {self.task_id}")
            self.change_task()

    def _increment_epoch_count(self):
        self.total_epoch += 1


class StreamClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.task_id = 0
        self.total_epoch = 0
        self.task_num = config["task_num"] if "task_num" in config else 1

    def change_task(self):
        self.fl_train_ds.change_idxs(list(self.index_list)[self.task_id::self.task_num])

    def receive_notify(self):
        super().receive_notify()
        task_id = self.message_queue.get_from_downlink(self.client_id, "task_id")
        if task_id != self.task_id:
            self.task_id = task_id
            print(f"Client {self.client_id} changed task to {self.task_id}")
            self.change_task()


class StreamClientWithGlobal(StreamClient):
    """
    StreamClientWithGlobal is a class that inherits from StreamClient and is used to implement the stream data
    scenario. This class's data is controlled by the server, and the model is trained on each task in turn.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.task_num = len(index_list)

    def change_task(self):
        self.fl_train_ds.change_idxs(self.index_list[self.task_id])
        # self.fl_train_ds = FLDataset(self.train_ds, list(self.index_list[self.task_id]), self.transform,
        #                              self.target_transform)
        # self.train_dl = DataLoader(self.fl_train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.task_id = (self.task_id + 1) % self.task_num


class StreamClientWithDir(StreamClient):
    """
    StreamClientWithDir is a class that inherits from StreamClient and is used to implement the stream data
    scenario. This class's data is divided into multiple tasks which customized by the user, and the model is trained on
    each task in turn.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.task_index_list = [[] for _ in range(self.task_num)]
        self.previous_model = {}

    def init_client(self):
        super().init_client()
        iid_config = self.config["iid"]
        train_labels = self.train_ds.targets[self.index_list]
        if isinstance(iid_config, bool):
            index_list = generate_iid_data(train_labels, self.task_num)
        elif isinstance(iid_config, dict) and "path" in iid_config:
            data_distribution_generator = ModuleFindTool.find_class_by_path(iid_config["path"])(iid_config["params"])
            index_list = data_distribution_generator.generate_data(iid_config, train_labels, self.task_num,
                                                                   self.fl_train_ds)
        else:
            index_list = generate_non_iid_data(iid_config, train_labels, self.task_num)
        self.task_index_list = index_list
        self.print_data_distribution()

    def print_data_distribution(self):
        for i in range(self.task_num):
            labels = self.train_ds.targets[self.index_list]
            label_counts = torch.bincount(labels[self.task_index_list[i]])
            print(f"Client {self.client_id}, Task {i}: {len(self.task_index_list[i])}, {label_counts}")

    def change_task(self):
        self.fl_train_ds.change_idxs(self.index_list[self.task_index_list[self.task_id]])
        # self.fl_train_ds = FLDataset(self.train_ds, self.index_list[self.task_index_list[self.task_id]], self.transform,
        #                              self.target_transform)
        # self.train_dl = DataLoader(self.fl_train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.task_id = (self.task_id + 1) % self.task_num


class ContinualClient(StreamClientWithGlobal):
    """
    ContinualClient is a class that inherits from StreamClientWithGlobal and is used to implement the continual learning
    scenario. In this scenario, client trains on different independent datasets in sequence.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.label_mapping = None

    def init_client(self):
        super().init_client()
        self.label_mapping = self.message_queue.get_from_downlink(self.client_id, "label_mapping")

    def train_one_epoch(self):
        if self.mu != 0:
            global_model = deepcopy(self.model)
        data_sum = 0
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                if self.label_mapping is not None:
                    label = torch.LongTensor([self.label_mapping[i.item()] for i in label])
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                # Calculate the loss function
                loss = self.loss_func(preds, label)
                data_sum += label.size(0)
                # proximal term
                if self.mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (self.mu / 2) * proximal_term
                # backpropagate
                loss.backward()
                # Update the gradient
                self.opti.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # Zero out the gradient and initialize the gradient.
                self.opti.zero_grad()
        # Return the updated model parameters obtained by training on the client's own data.
        weights = self.model.state_dict()
        torch.cuda.empty_cache()
        return data_sum, weights


class ContinualClientWithEWC(ContinualClient):
    """
    ContinualClientWithEWC is a class that inherits from ContinualClient and is used to implement the EWC loss.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.previous_model = None
        self.fisher = None
        self.ewc_lambda = 2000

    def receive_notify(self):
        if self.total_epoch % self.task_interval == 0:
            if self.fisher is None and self.task_id == 0:
                pass
            else:
                del self.fisher
                del self.previous_model
                self.previous_model = {n: deepcopy(p.data) for n, p in self.model.named_parameters() if p.requires_grad}
                self.compute_fisher(self.train_dl)
        super().receive_notify()

    def train_one_epoch(self):
        if self.mu != 0:
            global_model = deepcopy(self.model)
        data_sum = 0
        self.model.train()
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
                if self.label_mapping is not None:
                    label = torch.LongTensor([self.label_mapping[i.item()] for i in label])
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.model(data)
                # Calculate the loss function
                loss = self.loss_func(preds, label)
                data_sum += label.size(0)
                # proximal term
                if self.mu != 0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss = loss + (self.mu / 2) * proximal_term
                if self.fisher is not None:
                    loss += self.ewc_loss()
                # backpropagate
                loss.backward()
                # Update the gradient
                self.opti.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # Zero out the gradient and initialize the gradient.
                self.opti.zero_grad()
        # Return the updated model parameters obtained by training on the client's own data.
        weights = self.model.state_dict()
        torch.cuda.empty_cache()
        return data_sum, weights

    def compute_fisher(self, data_loader, num_samples=512):
        """
        Compute the Fisher information matrix for the current model

        Args:
            data_loader: PyTorch DataLoader for the training data
            num_samples: number of samples to use for computing the Fisher matrix
        """
        fisher = {}
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in deepcopy(params).items():
            fisher[n] = torch.zeros_like(p.data, requires_grad=False)
        self.model.eval()
        # 计算fisher矩阵
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= num_samples:
                break
            inputs, targets = inputs.to(self.dev), targets.to(self.dev)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                fisher[name].data += param.grad.data.pow(2) / num_samples
        # 使用fisher矩阵计算ewc loss
        self.fisher = {n: p for n, p in fisher.items()}

    def ewc_loss(self):
        """
        Calculate the EWC loss based on the Fisher information matrix

        Returns:
            EWC loss: a scalar value representing the EWC loss
        """
        loss = 0
        for name, param in self.model.named_parameters():
            _loss = torch.sum(self.fisher[name] * (param - self.previous_model[name]).pow(2))
            loss += _loss.sum()
        loss = self.ewc_lambda / 2 * loss
        return loss
