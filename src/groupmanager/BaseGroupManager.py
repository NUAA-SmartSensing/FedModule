from group.GroupCaller import GroupCaller
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter


class BaseGroupManager:
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter().get()
        self.client_list = self.global_var['client_manager'].client_thread_list
        self.latency_list = self.global_var['client_manager'].client_staleness_list
        self.network_list = []
        self.group_client_num_list = []
        self.group_method = ModuleFindTool.find_class_by_path(self.config["group_method"]["path"])(self, self.config["group_method"]["params"])
        self.group_caller = GroupCaller(self)
        self.group_list, self.group_num = self.group_caller.group(self.client_list, self.latency_list)
        self.epoch_list = [0 for _ in range(self.global_var['global_config']['client_num'])]

    def group(self, client_list, latency_list):
        return self.group_caller.group(client_list, latency_list)

    def get_group_num(self):
        return self.group_num

    def get_group_list(self):
        return self.group_list

    def check_update(self):
        if self.group_method.check_update():
            self.group(self.client_list, self.latency_list)
            return True
        else:
            return False
