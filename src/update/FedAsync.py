from fedasync import UpdaterThread
import copy
import torch


class FedAsync:
    def update_server_weights(self, updater_thread: UpdaterThread, c_id, client_weights, data_sum, time_stamp, epoch, update_param):
        # b = 60
        # a = 0.1
        # alpha = 0.1
        # r = 1
        b = update_param["b"]
        a = update_param["a"]
        alpha = update_param["alpha"]
        r = update_param["r"]
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
