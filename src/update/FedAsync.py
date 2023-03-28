from fedasync import UpdaterThread
import copy
import torch


class FedAsync:
    def __init__(self, config):
        self.config = config

    def update_server_weights(self, updater_thread: UpdaterThread, epoch, update_list):
        update_dict = update_list[0]
        client_weights = update_dict["weights"]
        time_stamp = update_dict["time_stamp"]
        b = self.config["b"]
        a = self.config["a"]
        alpha = self.config["alpha"]
        r = self.config["r"]
        if (updater_thread.current_time.get_time() - time_stamp) <= b:
            s = 1
        else:
            s = float(1 / ((a * (updater_thread.current_time.get_time() - time_stamp - b)) + 1))

        alpha = alpha * s * r
        updated_parameters = {}
        server_weights = copy.deepcopy(updater_thread.server_network.state_dict())

        for key, var in client_weights.items():
            updated_parameters[key] = var.clone()
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
        for key, var in server_weights.items():
            updated_parameters[key] = (alpha * updated_parameters[key] + (1 - alpha) * server_weights[key])
        return updated_parameters
