import copy

import torch

from fedasync import UpdaterThread


class AsyncAvg:
    def __init__(self, config, updater_thread):
        self.config = config
        self.updater_thread = updater_thread

    def update_server_weights(self, epoch, update_list):
        update_dict = update_list[0]
        client_weights = update_dict["weights"]
        updated_parameters = {}
        server_weights = copy.deepcopy(self.updater_thread.server_network.state_dict())

        for key, var in client_weights.items():
            updated_parameters[key] = var.clone()
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        for key, var in server_weights.items():
            updated_parameters[key] = (updated_parameters[key] + server_weights[key]) / 2
        return updated_parameters
