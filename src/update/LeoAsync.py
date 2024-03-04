import copy

import torch

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class LeoAsync(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter.get()

    def update_server_weights(self, epoch, update_list):
        
        group_ready_num = self.global_var["queue_manager"].group_ready_num
        epoch_list = self.global_var["group_manager"].epoch_list

        update_dict = update_list[group_ready_num]
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
        return updated_parameters, updated_parameters
