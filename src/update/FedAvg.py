import copy

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class FedAvg(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter().get()

    def update_server_weights(self, epoch, update_list):
        total_nums = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * update_list[0]["data_sum"] / total_nums
        for i in range(len(update_list) - 1):
            update_dict = update_list[i + 1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] * update_dict["data_sum"] / total_nums
        return updated_parameters, updated_parameters
