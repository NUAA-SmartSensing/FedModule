from clientmanager.BaseClientManager import BaseClientManager
from core.Runtime import CLIENT_STATUS
from utils import ModuleFindTool
from utils.GlobalVarGetter import GlobalVarGetter
from core.MessageQueue import EventFactory


class NormalClientManager(BaseClientManager):
    def __init__(self, whole_config):
        super().__init__(whole_config)
        self.global_var = GlobalVarGetter.get()
        self.client_list = []  # client list
        self.client_id_list = []  # client id list
        self.client_status = []  # client status list

        self.multi_gpu = whole_config["global"]["multi_gpu"]
        self.total_client_num = whole_config["global"]["client_num"]
        self.client_num = whole_config["client_manager"]["init_client_num"] if "init_client_num" in whole_config[
            "client_manager"] else self.total_client_num
        self.client_staleness_list = whole_config["client_manager"]["stale_list"]
        self.index_list = whole_config["client_manager"]["index_list"]  # each client's index list
        self.client_config = whole_config["client"]

        self.client_class = ModuleFindTool.find_class_by_path(whole_config["client"]["path"])
        self.stop_event_list = [EventFactory.create_Event() for _ in range(self.client_num)]
        self.selected_event_list = [EventFactory.create_Event() for _ in range(self.client_num)]
        self.global_var['selected_event_list'] = self.selected_event_list

    def start_all_clients(self):
        self.__init_clients()
        # 启动clients
        self.global_var['client_list'] = self.client_list
        self.global_var['client_id_list'] = self.client_id_list
        print("Start clients:")
        for i in self.client_id_list:
            self.client_list[i].start()
            self.client_status[i] = CLIENT_STATUS['active']

    def __init_clients(self):
        self.client_dev = self.get_client_dev_list(self.total_client_num, self.multi_gpu)
        for i in range(self.client_num):
            client_delay = self.client_staleness_list[i]
            self.client_list.append(
                self.client_class(i, self.stop_event_list[i], self.selected_event_list[i], client_delay,
                                  self.index_list[i], self.client_config, self.client_dev[i]))  # 实例化
            self.client_status.append(CLIENT_STATUS['created'])
            self.client_id_list.append(i)

    def get_client_list(self):
        return self.client_list

    def get_client_id_list(self):
        return self.client_id_list

    def stop_all_clients(self):
        # 终止所有client线程
        for i in self.client_id_list:
            self.stop_client_by_id(i)
            self.client_status[i] = CLIENT_STATUS['exited']

    def stop_client_by_id(self, client_id):
        self.stop_event_list[client_id].set()
        self.selected_event_list[client_id].set()
        self.client_status[client_id] = CLIENT_STATUS['exited']

    def create_and_start_new_client(self, dev='cpu'):
        client_id = len(self.client_list)
        self.client_list.append(
            self.client_class(client_id, self.stop_event_list[client_id], self.selected_event_list[client_id],
                              self.client_staleness_list[client_id], self.index_list[client_id], self.client_config, self.client_dev[client_id]))  # 实例化
        self.client_id_list.append(client_id)
        self.client_list[client_id].start()
        self.client_status.append(CLIENT_STATUS['active'])
        self.client_num += 1

    def client_join(self):
        for i in self.client_list:
            i.join()
