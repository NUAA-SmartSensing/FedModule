import copy

from update.AbstractUpdate import AbstractUpdate
from utils.GlobalVarGetter import GlobalVarGetter


class TWAFL(AbstractUpdate):
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter().get()
        self.flag = False
        self.items = None

    def update_server_weights(self, epoch, update_list):
        if self.items is None:
            # 初始化
            self.items = []
            for k in update_list[0]["weights"].keys():
                if "fc" in k:
                    self.items.append(k)
        self.flag = epoch % 15 in [11, 13, 14, 12, 0]
        if self.flag:
            server_weights = copy.deepcopy(self.global_var['updater'].model.state_dict())
        total_nums = 0
        for update_dict in update_list:
            total_nums += update_dict["data_sum"]
        updated_parameters = {}
        for key, var in update_list[0]["weights"].items():
            if self.flag and key not in self.items:
                updated_parameters[key] = server_weights[key]
            else:
                updated_parameters[key] = update_list[0]["weights"][key] *  update_list[0]["data_sum"] / total_nums
        for i in range(len(update_list) - 1):
            update_dict = update_list[i + 1]
            client_weights = update_dict["weights"]
            for key, var in client_weights.items():
                if self.flag and key not in self.items:
                    pass
                else:
                    updated_parameters[key] += client_weights[key] * update_dict["data_sum"] / total_nums
        return updated_parameters, None
