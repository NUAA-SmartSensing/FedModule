from utils import ModuleFindTool


class Grouping:
    def __init__(self, config):
        self.config = config
        self.client_list = []
        self.latency_list = []
        self.group_list = []
        self.group_num = 0

    def init(self, client_list, latency_list):
        self.client_list = client_list
        self.latency_list = latency_list
        self.group_list, self.group_num = self.group(client_list, latency_list, self.config)

    def group(self, client_list, latency_list, config):
        return self.group_list, self.group_num

    def get_group_num(self):
        return self.group_num

    def get_group_list(self):
        return self.group_list
