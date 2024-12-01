import copy

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter
import torch


class FedNova(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter().get()

    def update_server_weights(self, epoch, update_list):
        total_nums = 0
        total_tau = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
            total_tau += update_dict["data_sum"] * update_dict["tau"]
        pre_param = total_tau / total_nums
        server_weights = copy.deepcopy(self.global_var['updater'].model.state_dict())
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * update_list[0]["data_sum"] / (total_nums * update_list[0]["tau"])
        for i in range(len(update_list) - 1):
            update_dict = update_list[i + 1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] * update_dict["data_sum"] / (total_nums * update_dict["tau"])
        for key, var in updated_parameters.items():
            if torch.cuda.is_available():
                updated_parameters[key] = updated_parameters[key].cuda()
            updated_parameters[key] = server_weights[key] + updated_parameters[key] * pre_param
        return updated_parameters, None
