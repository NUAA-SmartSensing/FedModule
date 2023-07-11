import random

from client import NormalClient


class ProxClient(NormalClient.NormalClient):
    def train(self):
        if self.config["params"]["straggler"]:
            self.epoch = random.randint(1, self.config["params"]["local_epoch"])
        return self.train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func, self.opti, self.mu)
