import copy

from core.handlers.Handler import Handler
from utils import ModuleFindTool
from utils.Tools import to_cpu


class ClientTrainHandler(Handler):
    def _handle(self, request):
        client = request.get('client')
        config = client.config
        if 'train_func' in config:
            train_func = ModuleFindTool.find_class_by_path(config['train_func'])
        else:
            train_func = BasicTrain
        request['train_res'] = train_func(client.train_dl, client.model, client.loss_func, client.optimizer, client.epoch,
                                          client.dev,
                                          client.lr_scheduler, client.mu)
        return request


class PostTrainHandler(Handler):
    def _handle(self, request):
        client = request.get('client')
        weights, data_sum = request.get('train_res')
        client.model.load_state_dict(weights)
        client.upload_item('weights', to_cpu(weights))
        client.upload_item('data_sum', data_sum)
        return request


def BasicTrain(train_dl, model, loss_func, opti, epoch, dev, lr_scheduler=None, mu=0):
    raw_model = copy.deepcopy(model)
    model.train()
    data_sum = 0
    for epoch in range(epoch):
        for data, label in train_dl:
            opti.zero_grad()
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
            opti.step()
        if lr_scheduler:
            lr_scheduler.step()
    weights = model.state_dict()
    return weights, data_sum
