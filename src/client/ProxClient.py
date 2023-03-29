import random

from client import NormalClient
from utils.ModelTraining import train_one_epoch


class ProxClient(NormalClient.NormalClient):
    def train_one_epoch(self, r_weights):
        if self.config["params"]["straggler"]:
            self.epoch = random.randint(1, self.config["params"]["local_epoch"])
        return train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func, self.opti, self.mu)
