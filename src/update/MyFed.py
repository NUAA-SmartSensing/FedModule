from fedasync import UpdaterThread
import copy
import torch


class MyFed:
    def __init__(self):
        self.total_data = 0
        self.total_quality = 0

    def update_server_weights(self, updater_thread: UpdaterThread, epoch, update_dict, update_param):
        client_weights = update_dict["weights"]
        data_sum = update_dict["data_sum"]
        time_stamp = update_dict["time_stamp"]
        self.total_data += data_sum
        b = update_param["b"]
        a = update_param["a"]
        alpha = update_param["alpha"]
        r = update_param["r"]
        c = 1
        q = 1

        if (updater_thread.current_time.get_time() - time_stamp) <= b:
            s = 1
        else:
            s = float(1 / ((a * (updater_thread.current_time.get_time() - time_stamp - b)) + 1))

        alpha = alpha * s * r
        server_weights = copy.deepcopy(updater_thread.server_network.state_dict())

        total_diff = 0
        for key, var in client_weights.items():
            total_diff += torch.sum((server_weights[key] - client_weights[key]) ** 2)
        total_diff = total_diff.tolist()
        self.total_quality += total_diff
        if (updater_thread.current_time.get_time() - time_stamp) > b:
            if self.total_data <= (epoch + 1) * data_sum:
                c = 2 - self.total_data / ((epoch + 1) * data_sum)
            if self.total_quality <= (epoch + 1) * total_diff:
                q = 2 - self.total_quality / ((epoch + 1) * total_diff)
        reward = c * q

        updated_parameters = {}
        for key, var in client_weights.items():
            updated_parameters[key] = var.clone()
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()

        for key, var in server_weights.items():
            updated_parameters[key] = (alpha * reward * updated_parameters[key] + (1 - alpha) * server_weights[key])
        return updated_parameters
