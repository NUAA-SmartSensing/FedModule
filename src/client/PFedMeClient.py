import copy

from torch.optim import Optimizer

from client.TestClient import TestClient


class PFedMeClientRaw(TestClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        super().__init__(c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.K = config["K"] if "K" in config else 30
        self.lamda = config["optimizer"]["params"]["lamda"] if "lamda" in config["optimizer"]["params"] else 0.1
        self.learning_rate = config["optimizer"]["params"]["lr"] if "lr" in config["optimizer"]["params"] else 0.01

    def train_one_epoch(self):
        data_sum = 0
        global_model = copy.deepcopy(list(self.model.parameters()))
        for _ in range(self.epoch):  # local update
            X, y = self.get_next_train_batch()
            data_sum += y.size(0)
            for i in range(self.K):
                self.opti.zero_grad()
                output = self.model(X)
                loss = self.loss_func(output, y)
                loss.backward()
                persionalized_model_bar, _ = self.opti.step(global_model)

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(persionalized_model_bar, global_model):
                localweight.data = localweight.data - self.lamda * self.learning_rate * (
                            localweight.data - new_param.data)
            for param, new_param in zip(self.model.parameters(), global_model):
                param.data = new_param.data.clone()
        data_sum = 1
        return data_sum, self.model.state_dict()

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.train_dl)
            (X, y) = next(self.iter_trainloader)
        return X.to(self.dev), y.to(self.dev)

    def init_client(self):
        super().init_client()
        self.iter_trainloader = iter(self.train_dl)


class PFedMeClient(PFedMeClientRaw):
    def train_one_epoch(self):
        data_sum = 0
        global_model = copy.deepcopy(list(self.model.parameters()))
        for _ in range(self.epoch):  # local update
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                data_sum += label.size(0)
                for i in range(self.K):
                    self.opti.zero_grad()
                    output = self.model(data)
                    loss = self.loss_func(output, label)
                    loss.backward()
                    persionalized_model_bar, _ = self.opti.step(global_model)

                # update local weight after finding aproximate theta
                for new_param, localweight in zip(persionalized_model_bar, global_model):
                    localweight.data = localweight.data - self.lamda * self.learning_rate * (
                                localweight.data - new_param.data)
                for param, new_param in zip(self.model.parameters(), global_model):
                    param.data = new_param.data.clone()
        return data_sum, self.model.state_dict()


class PFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=15, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(PFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
        return group['params'], loss
