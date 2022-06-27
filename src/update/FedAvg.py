from fedasync import UpdaterThread
import copy
import torch


class FedAvg:
    def update_server_weights(self, updater_thread: UpdaterThread, epoch, update_list, update_param):
        client_nums = len(update_list)
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"]
        for i in range(len(update_list)-1):
            update_dict = update_list[i+1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key]
        for key, var in updated_parameters.items():
            updated_parameters[key] = updated_parameters[key] / client_nums
        return updated_parameters
