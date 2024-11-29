import torch
from torch.utils.data import DataLoader

from core.handlers.Handler import HandlerChain, Handler
from loss.LossFactory import LossFactory
from utils import ModuleFindTool
from utils.DatasetUtils import FLDataset
from utils.Tools import random_seed_set


class InitHandler(HandlerChain):
    def __init__(self):
        super().__init__()
        self._head = RandomSeedInit()
        (self._head.set_next(TrainDatasetInit())
         .set_next(TestDatasetInit())
         .set_next(ModelInit())
         .set_next(LossInit())
         .set_next(OptimizerInit())
         .set_next(SchedulerInit())
         .set_next(RegisterHandler())
         )


class RandomSeedInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        random_seed_set(config["seed"])
        return request


class TrainDatasetInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        client = request.get('client')
        client.train_ds = client.message_queue.get_train_dataset()
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
        return request


class ModelInit(Handler):
    def _handle(self, request):
        config = request.get('config')
        client = request.get('client')
        client.model = self._get_model(config)
        client.model = client.model.to(client.dev)
        client.training_params = {k: False for k in client.model.state_dict()}
        for n, p in client.model.named_parameters():
            client.training_params[n] = p.requires_grad
        return request

    @staticmethod
    def _get_model(config):
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
