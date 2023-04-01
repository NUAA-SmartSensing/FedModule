from fedsemi import UpdaterThread


class FedAT:
    def __init__(self, config, updater_thread):
        self.config = config
        self.updater_thread = updater_thread

    def update_server_weights(self, epoch, update_list):
        group_num = len(update_list)
        epoch_list = self.updater_thread.get_epoch_list()
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            updated_parameters[key] = update_list[0]["weights"][key] * epoch_list[group_num - 1 - 0] / epoch
        for i in range(len(update_list) - 1):
            update_dict = update_list[i + 1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                updated_parameters[key] += client_weights[key] * epoch_list[group_num - 1 - i] / epoch
        # 下发给客户端的权重
        self.updater_thread.global_var['scheduler'].server_weights = updated_parameters
        return updated_parameters
