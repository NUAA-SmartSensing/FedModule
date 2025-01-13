import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.handlers.Handler import HandlerChain, Handler
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset
from utils.ModuleFindTool import load_model_from_config
from utils.Tools import random_seed_set


class InitHandler(HandlerChain):
    def __init__(self):
        super().__init__()
        self._head = RandomSeedInit()
        (self._head.set_next(DatasetLoader())
         .set_next(TrainDatasetInit())
         .set_next(ModelInit())
         .set_next(LossInit())
         .set_next(OptimizerInit())
         .set_next(SchedulerInit())
         .set_next(RegisterHandler())
         )


class InitHandlerWithTest(InitHandler):
    def __init__(self):
        super().__init__()
        self.add_handler_after(TestDatasetInit(), DatasetLoader)


class RandomSeedInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        random_seed_set(config["seed"])
        return request


class DatasetLoader(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.train_ds = client.message_queue.get_train_dataset()
        return request


class TrainDatasetInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        client = request.get('client')
        transform, target_transform = self._get_transform(config)
        client.fl_train_ds = FLDataset(client.train_ds, list(client.index_list), transform, target_transform)
        client.train_dl = DataLoader(client.fl_train_ds, batch_size=client.batch_size, shuffle=True, drop_last=True)
        return request

    def _get_transform(self, config):
        transform, target_transform = None, None
        if "transform" in config:
            transform_func = ModuleFindTool.find_class_by_path(config["transform"]["path"])
            transform = transform_func(**config["transform"]["params"])
        if "target_transform" in config:
            target_transform_func = ModuleFindTool.find_class_by_path(config["target_transform"]["path"])
            target_transform = target_transform_func(**config["target_transform"]["params"])
        return transform, target_transform


class TestDatasetInit(Handler):
    def _handle(self, request):
        client = request.get('client')
        config = request.get('config')
        test_size = config.get("test_size", 0.1)
        n1 = int(len(client.index_list) * test_size)
        test_batch_size = min(config.get("test_batch_size", 32), n1)
        client.test_index_list, client.index_list = self.__split_list(client.index_list, [n1])
        test_transform = client.test_transform if hasattr(client, 'test_transform') else client.transform
        test_target_transform = client.test_target_transform if hasattr(client,
                                                                        'test_target_transform') else client.target_transform
        client.fl_test_ds = FLDataset(client.train_ds, list(client.test_index_list), test_transform,
                                      test_target_transform)
        client.test_dl = DataLoader(client.fl_test_ds, batch_size=test_batch_size, shuffle=True, drop_last=True)
        return request

    @staticmethod
    def __split_list(lst, length):
        random.shuffle(lst)
        lst = np.array(lst)
        return np.split(lst, length)


class ModelInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        client = request.get('client')
        client.model = load_model_from_config(config.get('model'), client)
        client.model = client.model.to(client.dev)
        client.training_params = {k: False for k in client.model.state_dict()}
        for n, p in client.model.named_parameters():
            client.training_params[n] = p.requires_grad
        return request


class LossInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        client = request.get('client')
        client.loss_func = LossFactory.create_loss(config['loss'], client)
        return request


class OptimizerInit(Handler):
    def _handle(self, request):
        client = request.get('client')
        config = request.get('config')
        optimizer_config = config["optimizer"]
        path = optimizer_config.get("path")
        params = optimizer_config.get("params", {})
        parameters = client.model.parameters()
        client.optimizer = ModuleFindTool.find_class_by_path(path)(parameters, **params)
        return request


class SchedulerInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        scheduler_config = config.get("scheduler", {})
        if scheduler_config:
            client = request.get('client')
            path = scheduler_config.get("path")
            params = scheduler_config.get("params", {})
            lr_scheduler = ModuleFindTool.find_class_by_path(path)(client.opti, **params)
            client.lr_scheduler = lr_scheduler
        return request


class RegisterHandler(Handler):
    def _handle(self, request):
        client = request.get('client')
        client.message_queue.set_training_status(client.client_id, False)
        return request
