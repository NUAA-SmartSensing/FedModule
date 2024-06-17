import copy

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class FedAT(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter.get()

    def update_server_weights(self, epoch, update_list):
        group_num = self.global_var["group_manager"].group_num
        epoch_list = self.global_var["group_manager"].epoch_list
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * epoch_list[group_num - 1 - 0] / epoch
        for i in range(group_num - 1):
            update_dict = update_list[i + 1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] * epoch_list[group_num - 2 - i] / epoch
        return updated_parameters, None
