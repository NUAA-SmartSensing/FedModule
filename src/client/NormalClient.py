import copy
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from client.Client import Client
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset
from utils.Tools import to_cpu


class NormalClient(Client):
    r"""
    This class inherits from Client and implements a basic Client.

    Attributes:
    fl_train_ds: FLDataset
        Local training dataset
    opti: Object
        Optimizer
    loss_func: Object
        Loss function
    train_dl: torch.utils.data.DataLoader
        Training data loader
    batch_size: int
        Batch size
    epoch: int
        Epoch number
    optimizer_config: dict
        Optimizer configuration
    mu: float
        Regularization coefficient
    config: dict
        Configuration
    dev: str
        Device
    """

    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        Client.__init__(self, c_id, stop_event, selected_event, delay, index_list, dev)
        self.lr_scheduler = None
        self.fl_train_ds = None
        self.opti = None
        self.loss_func = None
        self.train_dl = None
        self.batch_size = config["batch_size"]
        self.epoch = config["epochs"]
        self.optimizer_config = config["optimizer"]
        self.mu = config["mu"]
        self.config = config

    def run(self):
        """
        The primary running function of Client is used for clients with a base class of process,
        which executes before being woken up by the server.
        """
        self.init_client()
        self.message_queue.set_training_status(self.client_id, False)
        while not self.stop_event.is_set():
            # The client is selected and starts local training.
            if self.event.is_set():
                self.event.clear()
                self.local_run()
            # The client waits to be selected.
            else:
                self.event.wait()

    def local_run(self):
        """
        The run function of Client runs the main body, suitable for use as a target parameter of process.
        """
        self.message_queue.set_training_status(self.client_id, True)
        self.receive_notify()
        self.local_task()
        self.message_queue.set_training_status(self.client_id, False)

    def local_task(self):
        """
        The local task of Client, namely, the detailed process of training a model.
        """
        # The client performs training.
        data_sum, weights = self.train()
        print("Client", self.client_id, "trained")

        # Information transmitted from the client to the server has latency.
        self.delay_simulate(self.delay)

        # upload its updates
        self.upload(data_sum, weights)

    def train(self):
        data_sum, weights = self.train_one_epoch()
        return data_sum, to_cpu(weights)

    def upload(self, data_sum, weights):
        """
        The detailed parameters uploaded to the server by Client.
        """
        update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                       "time_stamp": self.time_stamp}
        self.message_queue.put_into_uplink(update_dict)
        print("Client", self.client_id, "uploaded")

    def train_one_epoch(self):
        """
        The training function of Client, used for model training.
        """
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
        data_sum = 0
        for epoch in range(self.epoch):
            for data, label in self.train_dl:
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

    def receive_notify(self):
        """
        Receive server notifications,
        including whether model parameters are received and timestamp information, etc.
        """
        received_weights = False
        received_time_stamp = False
        while not received_weights:
            received_weights = copy.deepcopy(self.message_queue.get_from_downlink(self.client_id, 'received_weights'))
            time.sleep(0.1)
        self.message_queue.put_into_downlink(self.client_id, 'received_weights', False)
        weights_buffer = self.message_queue.get_from_downlink(self.client_id, 'weights_buffer')
        state_dict = self.model.state_dict()
        for k in weights_buffer:
            if self.training_params[k]:
                state_dict[k] = copy.deepcopy(weights_buffer[k])
        self.model.load_state_dict(state_dict)
        while not received_time_stamp:
            received_time_stamp = copy.deepcopy(self.message_queue.get_from_downlink(self.client_id, 'received_time_stamp'))
            time.sleep(0.1)
        self.message_queue.put_into_downlink(self.client_id, 'received_time_stamp', False)
        self.time_stamp = copy.deepcopy(self.message_queue.get_from_downlink(self.client_id, 'time_stamp_buffer'))
        self.schedule_t = copy.deepcopy(self.message_queue.get_from_downlink(self.client_id, 'schedule_time_stamp_buffer'))

    def init_client(self):
        config = self.config
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])

        self.train_ds = copy.deepcopy(self.message_queue.get_train_dataset())

        self.transform, self.target_transform = self._get_transform(config)
        self.fl_train_ds = FLDataset(self.train_ds, list(self.index_list), self.transform, self.target_transform)

        self.model = self._get_model(config)
        self.model = self.model.to(self.dev)
        self.training_params = {k: False for k in self.model.state_dict()}
        for n, p in self.model.named_parameters():
            self.training_params[n] = p.requires_grad

        # optimizer
        opti_class = ModuleFindTool.find_class_by_path(self.optimizer_config["path"])
        self.opti = opti_class(self.model.parameters(), **self.optimizer_config["params"])
        if "scheduler" in config:
            scheduler_class = ModuleFindTool.find_class_by_path(config["scheduler"]["path"])
            self.lr_scheduler = scheduler_class(self.opti, **config["scheduler"]["params"])

        # loss function
        self.loss_func = LossFactory(config["loss"], self).create_loss()

        self.train_dl = DataLoader(self.fl_train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def delay_simulate(self, secs):
        """
        Simulate network and computation delays.

        Parameters:
            secs: int
                Delay time
        """
        time.sleep(secs)

    def _get_transform(self, config):
        transform, target_transform = None, None
        if "transform" in config:
            transform_func = ModuleFindTool.find_class_by_path(config["transform"]["path"])
            transform = transform_func(**config["transform"]["params"])
        if "target_transform" in config:
            target_transform_func = ModuleFindTool.find_class_by_path(config["target_transform"]["path"])
            target_transform = target_transform_func(**config["target_transform"]["params"])
        return transform, target_transform

    def _get_model(self, config):
        # local model
        if isinstance(config["model"], dict):
            model_class = ModuleFindTool.find_class_by_path(config["model"]["path"])
            for k, v in config["model"]["params"].items():
                if isinstance(v, str):
                    config["model"]["params"][k] = eval(v)
            model = model_class(**config["model"]["params"])
        elif isinstance(config["model"], str):
            model = torch.load(config["model"])
        else:
            raise ValueError("model config error")
        return model


class NormalClientWithDelta(NormalClient):
    """
    This class inherits from NormalClient and implements the functionality of uploading
    the difference between model parameters before and after training, i.e.,
    uploads $delta = w_i^t - w^t$, where $w_i^t$ is the model parameter updated by the client locally,
    and $w^t$ is the model parameter before local iteration.
    """

    def train_one_epoch(self):
        """
        The training function of Client, used for model training.

        Returns:
            data_sum: int
                Total number of local training data
            weights: dict
                Local training-derived model parameter differences $weights = w_i^t - w^t$
        """
        global_model = copy.deepcopy(self.model.state_dict())
        data_sum, weights = super().train_one_epoch()
        for k in weights:
            weights[k] = weights[k] - global_model[k]
        torch.cuda.empty_cache()
        return data_sum, weights


class NormalClientWithGrad(NormalClient):
    """
    This class inherits from NormalClient and implements the cumulative gradient upload after training.
    """

    def init_client(self):
        super().init_client()
        del self.opti

    def train_one_epoch(self):
        """
        The train function of the client, used for training models.

        Returns:
            data_sum: int
                Total number of local training data
            accumulated_grads: list
                Accumulated gradients
        """
        if self.mu != 0:
            global_model = copy.deepcopy(self.model)
        data_sum = 0
        accumulated_grads = []  # Initialize the list of accumulated gradients

        # Traverse the training data.
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = self.model(data)
            # Calculate the loss function
            loss = self.loss_func(preds, label)
            data_sum += label.size(0)
            # Proximal term
            if self.mu != 0:
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = loss + (self.mu / 2) * proximal_term
            # Backpropagate, but do not execute optimization steps.
            loss.backward()
            # Accumulate gradients.
            accumulated_grads = [None if acc_grad is None else acc_grad + param.grad
                                 for acc_grad, param in zip(accumulated_grads, self.model.parameters())]

            # Zero out the gradient to prepare for the next iteration.
            self.model.zero_grad()
        # return accumulate gradients.
        torch.cuda.empty_cache()
        return data_sum, accumulated_grads
