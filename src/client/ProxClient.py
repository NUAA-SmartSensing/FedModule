import random

from client.NormalClient import NormalClient


class ProxClient(NormalClient):
    def train(self):
        if self.config["params"]["straggler"]:
            self.epoch = random.randint(1, self.config["params"]["local_epoch"])
        return super().train()
