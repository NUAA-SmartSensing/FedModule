import copy
from collections import OrderedDict

from core.handlers.Handler import Handler
from utils import ModuleFindTool
from utils.Tools import to_cpu


class ClientTrainHandler(Handler):
    def _handle(self, request):
        client = request.get('client')
        config = client.config
        if 'train_func' in config:
            train_func = ModuleFindTool.find_class_by_path(config['train_func'])
        elif hasattr(client, 'train'):
            request['train_res'] = client.train()
            return request
        else:
            train_func = BasicTrain
        request['train_res'] = train_func(client.train_dl, client.model, client.loss_func, client.optimizer,
                                          client.epoch,
                                          client.dev,
                                          client.lr_scheduler, client.mu, client)
        return request


class ClientPostTrainHandler(Handler):
    def _handle(self, request):
        client = request.get('client')
        train_res = request.get('train_res')
        if len(train_res) >= 2:
            data_sum, weights, *others = train_res
            if others:  # 如果others不为空
                client.upload_item('others', others)
        else:
            raise ValueError("train_res must contain at least 2 elements")
        client.upload_item('weights', to_cpu(weights))
        client.upload_item('data_sum', data_sum)
        return request


def BasicTrain(train_dl, model, loss_func, optimizer, epoch, dev, lr_scheduler=None, mu=0, obj=None):
    raw_model = copy.deepcopy(model)
    model.train()
    data_sum = 0
    for _ in range(epoch):
        for data, label in train_dl:
            optimizer.zero_grad()
            data, label = data.to(dev), label.to(dev)
            preds = model(data)
            loss = loss_func(preds, label)
            data_sum += label.size(0)
            if mu != 0:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), raw_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = loss + (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
    weights = model.state_dict()
    return data_sum, weights


def TrainWithGrad(train_dl, model, loss_func, optimizer, epoch, dev, lr_scheduler=None, mu=0, obj=None):
    model.train()
    data_sum = 0
    for data, label in train_dl:
        optimizer.zero_grad()
        data, label = data.to(dev), label.to(dev)
        preds = model(data)
        loss = loss_func(preds, label)
        data_sum += label.size(0)
        loss.backward()
        break
    gradients = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            gradients[name] = param.grad.clone().detach()
    return data_sum, gradients


def TrainWithDelta(train_dl, model, loss_func, optimizer, epoch, dev, lr_scheduler=None, mu=0, obj=None):
    raw_model = copy.deepcopy(model.state_dict())
    data_sum, weights = BasicTrain(train_dl, model, loss_func, optimizer, epoch, dev, lr_scheduler, mu, obj)
    delta = OrderedDict()
    for name, param in raw_model.items():
        delta[name] = weights[name] - param
    return data_sum, delta
