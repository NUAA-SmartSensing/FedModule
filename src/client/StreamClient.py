from collections import Counter
from copy import deepcopy

import torch
from torch.nn import functional as F

from client.NormalClient import NormalClient
from client.mixin.ClientHandler import UpdateReceiver
from core.handlers.Handler import Handler
from utils import ModuleFindTool
from utils.IID import generate_iid_data, generate_non_iid_data


class StreamUpdateReceiver(Handler):
    def _handle(self, request):
        client = request.get("client")
        task_id = client.message_queue.get_from_downlink(client.client_id, "task_id")
        if task_id != client.task_id:
            client.task_id = task_id
            print(f"Client {client.client_id} changed task to {client.task_id}")
            client.change_task()
        return request


class SelfStreamClient(NormalClient):
    """
    StreamClient is a class that inherits from NormalClient and is used to implement the stream data scenario.
    This class's data is divided into multiple tasks equally, and the model is trained on each task in turn.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.task_num = config.get("task_num", 1)
        self.task_interval = config.get("task_interval", 1)
        self.task_id = 0
        self.total_epoch = 0

    def change_task(self):
        self.fl_train_ds.change_idxs(list(self.index_list)[self.task_id::self.task_num])
        self.task_id = (self.task_id + 1) % self.task_num

    def create_handler_chain(self):
        def _check_and_change_task(request):
            client = request.get("client")
            if client.total_epoch % client.task_interval == 0:
                print(f"Client {client.client_id} changed task to {client.task_id}")
                client.change_task()
            return request

        def _increment_epoch_count(request):
            client = request.get("client")
            client.total_epoch += 1
            return request

        super().create_handler_chain()
        self.handler_chain.add_handler_before(_check_and_change_task, UpdateReceiver)
        self.handler_chain.add_handler(_increment_epoch_count)


class StreamClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.task_id = -1
        self.task_num = config.get("task_num", 1)

    def change_task(self):
        self.fl_train_ds.change_idxs(list(self.index_list)[self.task_id::self.task_num])

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_after(StreamUpdateReceiver(), UpdateReceiver)


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


class DataPartitionerHandler(Handler):
    def _handle(self, request):
        client = request.get("client")
        iid_config = client.config["iid"]
        train_labels = client.train_ds.targets[client.index_list].numpy()
        if isinstance(iid_config, bool):
            index_list = generate_iid_data(train_labels, client.task_num)
        elif isinstance(iid_config, dict) and "path" in iid_config:
            data_distribution_generator = ModuleFindTool.find_class_by_path(iid_config["path"])(iid_config["params"])
            index_list = data_distribution_generator.generate_data(iid_config, train_labels, client.task_num,
                                                                   client.fl_train_ds)
        else:
            index_list = generate_non_iid_data(iid_config, train_labels, client.task_num)
        client.task_index_list = index_list
        client.print_data_distribution()
        return request


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

    def create_handler_chain(self):
        super().create_handler_chain()
        self.init_chain.add_handler(DataPartitionerHandler())

    def print_data_distribution(self):
        for i in range(self.task_num):
            labels = self.train_ds.targets[self.index_list].numpy()
            total = len(labels[self.task_index_list[i]])
            counts = Counter(labels[self.task_index_list[i]])
            print(f"Client {self.client_id}, Task {i}: {total}, ", end='')
            for k, v in counts.items():
                print(f" {k}: {v}",end=',')
            print()

    def change_task(self):
        self.fl_train_ds.change_idxs(self.index_list[self.task_index_list[self.task_id]])


class LabelMappingHandler(Handler):
    def _handle(self, request):
        client = request.get("client")
        label_mapping = client.message_queue.get_from_downlink(client.client_id, "label_mapping")
        client.label_mapping = label_mapping
        return request


class ContinualClient(StreamClientWithGlobal):
    """
    ContinualClient is a class that inherits from StreamClientWithGlobal and is used to implement the continual learning
    scenario. In this scenario, client trains on different independent datasets in sequence.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.label_mapping = None

    def create_handler_chain(self):
        super().create_handler_chain()
        self.init_chain.add_handler(LabelMappingHandler())

    def train(self):
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
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # Zero out the gradient and initialize the gradient.
                self.optimizer.zero_grad()
        # Return the updated model parameters obtained by training on the client's own data.
        weights = self.model.state_dict()
        torch.cuda.empty_cache()
        return data_sum, weights


class FisherHandler(Handler):

    def _handle(self, request):
        client = request.get("client")
        if client.total_epoch % client.task_interval == 0:
            if client.fisher is None and client.task_id == 0:
                pass
            else:
                del client.fisher
                del client.previous_model
                client.previous_model = {n: deepcopy(p.data) for n, p in client.model.named_parameters() if
                                         p.requires_grad}
                client.compute_fisher(client.train_dl)
        return request


class ContinualClientWithEWC(ContinualClient):
    """
    ContinualClientWithEWC is a class that inherits from ContinualClient and is used to implement the EWC loss.
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.previous_model = None
        self.fisher = None
        self.ewc_lambda = 2000

    def create_handler_chain(self):
        super().create_handler_chain()
        self.handler_chain.add_handler_after(FisherHandler(), UpdateReceiver)

    def train(self):
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
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                # Zero out the gradient and initialize the gradient.
                self.optimizer.zero_grad()
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
