import copy

import torch

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class FedAsync(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter.get()

    def update_server_weights(self, epoch, update_list):
        update_dict = update_list[0]
        client_weights = update_dict["weights"]
        time_stamp = update_dict["time_stamp"]
        b = self.config["b"]
        a = self.config["a"]
        alpha = self.config["alpha"]
        r = self.config["r"]
        if (self.global_var['updater'].current_time.get_time() - time_stamp) <= b:
            s = 1
        else:
            s = float(1 / ((a * (self.global_var['updater'].current_time.get_time() - time_stamp - b)) + 1))

        alpha = alpha * s * r
        updated_parameters = {}
        server_weights = self.global_var['updater'].server_network.state_dict()
        for key, var in server_weights.items():
            updated_parameters[key] = (alpha * client_weights[key] + (1 - alpha) * server_weights[key])
        return updated_parameters, None
