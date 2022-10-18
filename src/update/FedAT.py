from fedsemi import UpdaterThread
import copy
import torch


class FedAT:
    def __init__(self, config):
        self.config = config

    def update_server_weights(self, updater_thread: UpdaterThread, epoch, update_list):
        group_num = len(update_list)
        epoch_list = updater_thread.get_epoch_list()
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * epoch_list[group_num - 1 - 0] / epoch
        for i in range(len(update_list)-1):
            update_dict = update_list[i+1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] * epoch_list[group_num - 1 - i] / epoch
        return updated_parameters
