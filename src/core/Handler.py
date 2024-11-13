import warnings
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from utils.Tools import to_dev, to_cpu


class HandlerChain:
    def __init__(self, chain=None):
        self._head = chain

    def add_handler_after(self, handler, target_cls):
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            while it is not None:
                if isinstance(it, target_cls):
                    it.insert_next(handler)
                    break

    def add_handler_before(self, handler, target_cls):
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            if isinstance(it, target_cls):
                handler.insert_next(it)
                self._head = handler
            else:
                while it.next_handler is not None:
                    if isinstance(it.next_handler, target_cls):
                        handler.insert_next(it.next_handler)
                        it.insert_next(handler)
                        break
                    it = it.next_handler

    def exchange_handler(self, handler, target_cls):
        if self._head is None:
            warnings.warn("The head is None, the handler will be set as the head.")
            self._head = handler
        else:
            it = self._head
            if isinstance(it, target_cls):
                handler.insert_next(it.next_handler)
                self._head = handler
            else:
                while it.next_handler is not None:
                    if isinstance(it.next_handler, target_cls):
                        handler.insert_next(it.next_handler.next_handler)
                        it.insert_next(handler)
                        break
                    it = it.next_handler
    def handle(self, request):
        return self._head.handle(request)


class Handler:
    """抽象处理程序"""

    def __init__(self, handler=None):
        self.next_handler = handler

    def next(self, handler):
        self.next_handler = handler
        return self

    def insert_next(self, handler):
        if self.next_handler is None:
            warnings.warn("The next handler is None, the handler will be set as the next handler.")
            self.next_handler = handler
        else:
            handler.next_handler = self.next_handler
            self.next_handler = handler

    @abstractmethod
    def _handle(self, request):
        pass

    def handle(self, request):
        response = self._handle(request)
        if self.next_handler:
            return self.next_handler.handle(response)
        return response

    def __call__(self, *args, **kwargs):
        return self.handle(*args, **kwargs)


class ClientUpdateGetter(Handler):
    """更新获取处理程序"""

    def _handle(self, request):
        update_list = []
        global_var = request.get('global_var')
        queue_manager = global_var['queue_manager']
        # receive all updates
        while not queue_manager.empty():
            update_list.append(queue_manager.get())
        request['update_list'] = update_list
        return request


class Aggregation(Handler):
    """聚合处理程序"""

    def _handle(self, request):
        update_list = request.get('update_list')
        updater = request.get('updater')
        epoch = request.get('epoch')
        global_model, delivery_weights = updater.update_caller.update_server_weights(epoch, update_list)
        request['weights'] = global_model
        request['delivery_weights'] = delivery_weights
        return request


class GlobalModelOptimization(Handler):
    """全局模型优化处理程序"""

    def _handle(self, request):
        optim = request.get('optim')
        global_var = request.get('global_var')
        global_model = request.get('weights')
        updater = request.get('updater')

        new_model = to_dev(global_model, 'cuda')
        if optim is not None:
            training_params = global_var['training_params']
            old_model = updater.model.state_dict()
            g = {}
            for k in new_model:
                if training_params[k]:
                    g[k] = new_model[k] - old_model[k]
            for k, w in zip(g, updater.model.parameters()):
                w.grad = -g[k]
            optim.step()
        else:
            updater.model.load_state_dict(new_model)
        request['weights'] = to_cpu(updater.model.state_dict())
        return request


class ClientSelector(Handler):
    def _handle(self, request):
        return request


class ModelEvaluator(Handler):
    def _handle(self, request):
        updater = request.get('updater')
        epoch = request.get('epoch')
        dl = DataLoader(updater.test_data, batch_size=100, shuffle=True, drop_last=True)
        test_correct = 0
        test_loss = 0
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for data in dl:
                inputs, labels = data
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = updater.server_network(inputs)
                _, id = torch.max(outputs.data, 1)
                test_loss += updater.loss_func(outputs, labels).detach().item()
                test_correct += torch.sum(id == labels.data).cpu().numpy()
            accuracy = test_correct / len(dl)
            loss = test_loss / len(dl)
            updater.loss_list.append(loss)
            updater.accuracy_list.append(accuracy)
            print('Epoch(t):', epoch, 'accuracy:', accuracy, 'loss', loss)
        return request


class DispatchContentFilter(Handler):
    def _handle(self, request):
        delivery_weights = request.get('delivery_weights')
        new_global_model = request.get('weights')
        updater = request.get('updater')
        if delivery_weights is not None:
            updater.set_delivery_weights(delivery_weights)
        else:
            updater.set_delivery_weights(new_global_model)
        return request


class ServerMessageSender(Handler):
    def _handle(self, request):
        return request


class ClientUpdateSender(Handler):
    def _handle(self, request):
        return request
