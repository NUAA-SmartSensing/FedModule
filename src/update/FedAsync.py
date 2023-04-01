import copy

import torch

from fedasync import UpdaterThread


class FedAsync:
    def __init__(self, config, updater_thread):
        self.config = config
        self.updater_thread = updater_thread

    def update_server_weights(self, epoch, update_list):
        update_dict = update_list[0]
        client_weights = update_dict["weights"]
        time_stamp = update_dict["time_stamp"]
        b = self.config["b"]
        a = self.config["a"]
        alpha = self.config["alpha"]
        r = self.config["r"]
        if (self.updater_thread.current_time.get_time() - time_stamp) <= b:
            s = 1
        else:
            s = float(1 / ((a * (self.updater_thread.current_time.get_time() - time_stamp - b)) + 1))

        alpha = alpha * s * r
        updated_parameters = {}
        server_weights = copy.deepcopy(self.updater_thread.server_network.state_dict())

        for key, var in client_weights.items():
            updated_parameters[key] = var.clone()
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        for key, var in server_weights.items():
            updated_parameters[key] = (alpha * updated_parameters[key] + (1 - alpha) * server_weights[key])
        # 下发给客户端的权重
        self.updater_thread.global_var['scheduler'].server_weights = updated_parameters
        return updated_parameters
