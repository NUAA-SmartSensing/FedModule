from group.GroupCaller import GroupCaller
from groupmanager.BaseGroupManager import BaseGroupManager
from utils import ModuleFindTool


class NormalGroupManager(BaseGroupManager):
    def __init__(self, config):
        super().__init__(config)
        self.client_list = self.global_var['client_id_list']
        self.latency_list = self.global_var['client_staleness_list']
        self.network_list = []
        self.group_client_num_list = []
        self.group_method = ModuleFindTool.find_class_by_path(self.config["group_method"]["path"])(self, self.config[
            "group_method"]["params"])
        self.group_caller = GroupCaller(self)
        self.group_list, self.group_num = self.group_caller.group(self.client_list, self.latency_list)
        self.epoch_list = [0 for _ in range(self.global_var['global_config']['client_num'])]

    def __group(self, client_list, latency_list):
        return self.group_caller.group(client_list, latency_list)

    def get_group_num(self):
        return self.group_num

    def get_group_list(self):
        return self.group_list

    def update(self):
        self.__group(self.client_list, self.latency_list)
        return self.group_num
